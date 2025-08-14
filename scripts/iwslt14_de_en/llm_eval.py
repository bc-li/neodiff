#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from openai import OpenAI
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate MT quality using DeepSeek API")
    parser.add_argument("input_file", help="Path to input file in fairseq generate format")
    parser.add_argument("--output_dir", help="Custom output directory path")
    parser.add_argument("--detok_script", default="data/detokenizer.perl", 
                      help="Path to detokenizer Perl script. get it from https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl")

    parser.add_argument("--batch_size", type=int, default=32, help="Number of parallel evaluations")
    parser.add_argument("--max_retries", type=int, default=3, help="Max API retries per request")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API batches in seconds")
    parser.add_argument("--save_interval", type=int, default=1, 
                      help="Save results every N processed items (0 for real-time)")
    return parser.parse_args()

def run_detokenizer(script_path, input_file, output_file, lang="en"):
    """Run Perl detokenizer script with proper error handling"""
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            subprocess.run(
                ["perl", script_path, "-q", "-l", lang],
                stdin=infile,
                stdout=outfile,
                check=True
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Detokenization failed for {input_file}: {str(e)}")
        return False


def process_input_file(input_path, output_dir, detok_script):
    """Replicate COMET's preprocessing pipeline exactly"""
    # Create output directories
    detok_dir = Path(output_dir) / "detokenized"
    detok_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base filename
    filename = Path(input_path).stem
    
    # Stage 1: Extract segments (same as COMET)
    src_tok = detok_dir / f"{filename}.tok.src"
    ref_tok = detok_dir / f"{filename}.tok.ref"
    hyp_tok = detok_dir / f"{filename}.tok.sys"

    with open(input_path) as fin, \
         open(src_tok, 'w') as fsrc, \
         open(ref_tok, 'w') as fref, \
         open(hyp_tok, 'w') as fhyp:
        
        for line in fin:
            if line.startswith("S-"):
                fsrc.write(line.split("\t", 1)[1])
            elif line.startswith("T-"):
                fref.write(line.split("\t", 1)[1])
            elif line.startswith("H-"):
                fhyp.write(line.split("\t", 2)[2])

    # Stage 2: Detokenize (same as COMET)
    src_detok = detok_dir / f"{filename}.detok.src"
    ref_detok = detok_dir / f"{filename}.detok.ref"
    hyp_detok = detok_dir / f"{filename}.detok.sys"

    # Run detokenization for each file
    for lang, input_file, output_file in [
        ("en", src_tok, src_detok),
        ("de", ref_tok, ref_detok),
        ("de", hyp_tok, hyp_detok)
    ]:
        if not run_detokenizer(detok_script, input_file, output_file, lang):
            raise RuntimeError(f"Detokenization failed for {input_file}")

    # Create final output directory
    eval_dir = Path(output_dir) / "deepseek_eval"
    eval_dir.mkdir(exist_ok=True)

    # Copy detokenized files with COMET-style naming
    final_files = {
        src_detok: eval_dir / "src.en",
        ref_detok: eval_dir / "ref.de",
        hyp_detok: eval_dir / "hyp.de"
    }
    
    for src, dst in final_files.items():
        dst.write_text(src.read_text())
    
    return eval_dir


class DeepSeekEvaluator:
    def __init__(self, api_key, max_retries, delay):
        self.client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
        self.max_retries = max_retries
        self.delay = delay
#         self.prompt_template = """
# Evaluate this translation from {src_lang} to {tgt_lang} (0-100 score):  
# [Source] {source}  
# [Reference] {reference}  
# [Translation] {translation}  
# Score these aspects STRICTLY IN THIS ORDER:
# 1. **Accuracy**: Faithfulness to source meaning  
# 2. **Fluency**: Naturalness in target language  
# 3. **Completeness**: Information retention  
# 4. **Creativity**: Handling of ambiguous or open-ended source content  

# Return ONLY 4 numbers separated by commas, NO text. Example: 90,85,88,75
# """
        self.prompt_template = """
Evaluate this paraphrase generation (0-100 score):  
[Original] {source}  
[Reference] {reference}  
[Paraphrase] {translation}  
Score these aspects STRICTLY IN THIS ORDER:
1. **Semantic Faithfulness**: Meaning preservation from original
2. **Fluency**: Naturalness in language
3. **Completeness**: Retention of all information
4. **Phrasing Diversity**: Variation in wording/structure while preserving meaning

Return ONLY 4 numbers separated by commas, NO text. Example: 100,92,31,87
"""

    def get_score(self, source, reference, translation, lang_pair="en-de"):
        src_lang, tgt_lang = lang_pair.split("-")
        prompt = self.prompt_template.format(
            src_lang=src_lang.upper(),
            tgt_lang=tgt_lang.upper(),
            source=source,
            reference=reference,
            translation=translation
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20
                )
                
                score_str = response.choices[0].message.content.strip()
                scores = re.findall(r"\b(?:100|\d{1,2})\b", score_str)
                
                if len(scores) == 4:
                    return {
                        "Semantic Faithfulness": float(scores[0]),
                        "Fluency": float(scores[1]),
                        "Completeness": float(scores[2]),
                        "Phrasing Diversity": float(scores[3])
                    }
                
                print(f"Invalid response format: {score_str} (attempt {attempt+1})")
                sleep(self.delay * 2)
            
            except Exception as e:
                print(f"API Error: {str(e)} (attempt {attempt+1})")
                sleep(self.delay * 2)
        
        return None


class ResultWriter:
    def __init__(self, output_path: Path, args: argparse.Namespace):
        self.output_path = output_path
        self.args = args
        self.results = []
        self.dimension_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})
        
        # Initialize result file
        self._save_results(init=True)
    
    def add_result(self, score_dict: dict):
        self.results.append(score_dict)
        
        if score_dict is not None:
            for dim, score in score_dict.items():
                if score is not None:
                    self.dimension_stats[dim]["sum"] += score
                    self.dimension_stats[dim]["count"] += 1
        
        self._save_results()
    
    def _save_results(self, init=False):
        dimension_averages = {}
        for dim, stats in self.dimension_stats.items():
            if stats["count"] > 0:
                dimension_averages[dim] = stats["sum"] / stats["count"]
        
        valid_scores = [sum(s.values())/len(s) for s in self.results if s]
        global_avg = sum(valid_scores)/len(valid_scores) if valid_scores else 0
        
        result = {
            "system_scores": dimension_averages,
            "global_average": global_avg,
            "segment_details": self.results,
            "coverage": {dim: f"{stats['count']}/{len(self.results)}" 
                        for dim, stats in self.dimension_stats.items()},
            "parameters": vars(self.args)
        }
        
        temp_path = self.output_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        temp_path.replace(self.output_path)
        
        if init:
            print(f"Initialized result file at {self.output_path}")

def process_batch(evaluator, batch):    
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        futures = [
            executor.submit(evaluator.get_score, item["src"], item["ref"], item["hyp"])
            for item in batch
        ]
        return [future.result() for future in as_completed(futures)]

def main():
    args = parse_arguments()
    output_dir = process_input_file(
        args.input_file,
        args.output_dir or f"{Path(args.input_file).parent}/deepseek_output",
        args.detok_script
    )
    
    # Initialize evaluator
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY environment variable")
    
    evaluator = DeepSeekEvaluator(
        api_key=api_key,
        max_retries=args.max_retries,
        delay=args.delay
    )
    
    # Initialize result writer
    result_file = output_dir / "results.json"
    writer = ResultWriter(result_file, args)
    
    # Load processed data
    with open(output_dir/"src.en") as f:
        sources = [line.strip() for line in f]
    with open(output_dir/"ref.de") as f:
        references = [line.strip() for line in f]
    with open(output_dir/"hyp.de") as f:
        hypotheses = [line.strip() for line in f]

    total = len(sources)
    processed = 0

    # Batch processing with concurrency
    for batch_start in range(0, total, args.batch_size):
        batch_end = batch_start + args.batch_size
        batch_items = [
            (i, sources[i], references[i], hypotheses[i])
            for i in range(batch_start, min(batch_end, total))
        ]

        # Process current batch
        batch_results = []
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            future_to_item = {
                executor.submit(evaluator.get_score, src, ref, hyp): (i, src, ref, hyp)
                for i, src, ref, hyp in batch_items
            }
            
            for future in as_completed(future_to_item):
                i, src, ref, hyp = future_to_item[future]
                try:
                    score = future.result()
                    batch_results.append((i, score))
                except Exception as e:
                    print(f"Error processing item {i}: {str(e)}")
                    batch_results.append((i, None))

        # Sort results by original order and save
        for i, score in sorted(batch_results, key=lambda x: x[0]):
            writer.add_result(score)
        
        # Calculate batch statistics
        valid_scores = [
            sum(score.values())/4 
            for _, score in batch_results 
            if score and len(score)==4
        ]
        batch_avg = sum(valid_scores)/len(valid_scores) if valid_scores else 0
        valid_count = len(valid_scores)
        
        # Update progress
        processed += len(batch_items)
        print(
            f"Processed {processed}/{total} | "
            f"Batch avg: {batch_avg:.1f} | "
            f"Valid: {valid_count}/{len(batch_items)}".ljust(80), 
            end="\r"
        )

        # Inter-batch delay
        sleep(args.delay)
    
    # Final output
    print("\n\033[1;36m[Evaluation Complete]")
    
    # Calculate final statistics
    valid_scores = [
        sum(score.values())/len(score) 
        for score in writer.results 
        if score and len(score)==4
    ]
    
    if valid_scores:
        global_avg = sum(valid_scores)/len(valid_scores)
        valid_count = len(valid_scores)
        total = len(writer.results)
        
        print(f"\033[1;35mFinal Average Score : {global_avg:.2f}")
        print(f"\033[1;34mValid Segments      : {valid_count}/{total} ({valid_count/total:.1%})")
        
        # Display dimension averages
        print("\n\033[1;32mDimension Averages:")
        for dim in ['Semantic Faithfulness', 'Fluency', 'Completeness', 'Phrasing Diversity']:
            stats = writer.dimension_stats.get(dim, {'sum':0, 'count':0})
            if stats['count'] > 0:
                avg = stats['sum'] / stats['count']
                print(f"  {dim:<12}: {avg:.2f}")
    else:
        print("\033[1;31mNo valid scores available")
    
    print(f"\033[0mDetailed results saved to {result_file}")


if __name__ == "__main__":
    main()

