#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import argparse
import time
from datetime import datetime
from bayes_opt import BayesianOptimization


def _numeric_sort_key(item):
    key = item[0]
    try:
        return int(key[1:])
    except Exception:
        return key


class NeoDiffScheduleOptimizer:
    def __init__(self, model_name, dataset="iwslt14_de_en", device=0, batch_size=50, discrete=False, subset="valid"):
        self.model_name = model_name
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.discrete = discrete
        self.subset = subset
        self.model_dir = f"models/{dataset}/{model_name}"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("scripts", "iwslt14_de_en", "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"neodiff_optimization_{timestamp}.log")
        self.best_bleu = -100000
        self.best_timesteps = None
        self.total_iterations = 0
        self.start_time = None
        self.iteration_times = []
        self._prepare_checkpoints()

    def _prepare_checkpoints(self):
        last_avg_path = f"{self.model_dir}/ckpts/checkpoint_last_avg.pt"
        if not os.path.exists(last_avg_path):
            cmd = f"python scripts/average_checkpoints.py --inputs {self.model_dir}/ckpts/checkpoint[0-9]* --output {last_avg_path}"
            subprocess.run(cmd, shell=True)
        best_avg_path = f"{self.model_dir}/ckpts/checkpoint_best_avg.pt"
        if not os.path.exists(best_avg_path):
            cmd = f"python scripts/average_checkpoints.py --inputs {self.model_dir}/ckpts/checkpoint.* --output {best_avg_path}"
            subprocess.run(cmd, shell=True)

    def log_message(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def evaluate_schedule(self, timesteps, steps=20, method="normal", early_stopping=0):
        scheduler = "poisson"
        nf = 1
        length_beam = 9
        noise_beam = 1
        timesteps_str = ",".join(map(str, timesteps))
        output_name = f"optimize_step{steps}_beam{length_beam}x{noise_beam}_{method}_stop{early_stopping}_nf{nf}_timesteps{len(timesteps)}"
        suffix = "discrete" if self.discrete else "continuous"
        output_dir = f"{self.model_dir}/evaluate/{output_name}_{suffix}"
        os.makedirs(output_dir, exist_ok=True)
        self.log_message(f"[INFERENCE] Starting evaluation with timesteps: {timesteps}")
        self.log_message(f"[INFERENCE] Output directory: {output_dir}")
        ckpt_list = ["best_avg"]
        bleu_scores = {}
        for ckpt in ckpt_list:
            self.log_message(f"[INFERENCE] Running fairseq-generate for checkpoint: {ckpt}")
            cmd = f"""CUDA_VISIBLE_DEVICES={self.device} fairseq-generate data-bin/{self.dataset} \
                --gen-subset {self.subset} \
                --user-dir neodiff \
                --task neodiff_tuning_global_t \
                --path {self.model_dir}/ckpts/checkpoint_{ckpt}.pt \
                --decoding-steps {steps} \
                --decoding-early-stopping {early_stopping} \
                --decoding-scheduler {scheduler} \
                --decoding-method {method} \
                --decoding-nf {nf} \
                --length-beam-size {length_beam} \
                --noise-beam-size {noise_beam} \
                --bleu-mbr \
                --remove-bpe \
                --batch-size {self.batch_size} \
                --beam 9 \
                --retain-iter-history \
                --retain-z-0-hat \
                --uses-ema \
                --custom-timesteps {timesteps_str} \
                > {output_dir}/{ckpt}.txt"""
            self.log_message(f"[INFERENCE] Running fairseq-generate (progress bar will be shown)...")
            result = subprocess.run(cmd, shell=True)
            with open(f"{output_dir}/bleu.txt", "a") as f:
                f.write(f"{ckpt}\n")
                with open(f"{output_dir}/{ckpt}.txt") as result_file:
                    last_line = result_file.readlines()[-1]
                    f.write(last_line)
                    f.write("\n")
            bleu_score = -1.0
            try:
                with open(f"{output_dir}/{ckpt}.txt") as result_file:
                    last_line = result_file.readlines()[-1]
                    if "BLEU4" in last_line:
                        bleu_score = float(last_line.split("BLEU4 = ")[1].split(",")[0])
                    bleu_scores[ckpt] = bleu_score
                    self.log_message(f"[INFERENCE] Extracted BLEU score for {ckpt}: {bleu_score:.4f}")
            except Exception as e:
                self.log_message(f"[INFERENCE] Error extracting BLEU score for {ckpt}: {e}")
                bleu_scores[ckpt] = -1.0
        self.log_message(f"[INFERENCE] Finished evaluation. BLEU scores: {bleu_scores}")
        return bleu_scores

    def fitness_function(self, timesteps):
        current_bleu = self.evaluate_schedule(timesteps)["best_avg"]
        previous_best = self.best_bleu
        if current_bleu > self.best_bleu:
            self.best_bleu = current_bleu
            self.best_timesteps = timesteps
            self.log_message(f"[OPTIMIZATION] NEW BEST! BLEU: {current_bleu:.4f} (Previous: {previous_best:.4f})")
        log_msg = f"[OPTIMIZATION] Current BLEU: {current_bleu:.4f} | Best BLEU: {self.best_bleu:.4f} | Best timesteps: {self.best_timesteps}"
        self.log_message(log_msg)
        self.log_message("="*80)
        return current_bleu

    def target_function(self, **kwargs):
        iteration_start = time.time()
        if self.start_time is None:
            self.start_time = iteration_start
        self.total_iterations += 1
        timesteps_raw = [value for key, value in sorted(kwargs.items(), key=_numeric_sort_key)]
        if self.discrete:
            timesteps = [round(i * 10) / 10.0 for i in timesteps_raw]
        else:
            timesteps = timesteps_raw
        if hasattr(self, 'total_evaluations'):
            progress_pct = (self.total_iterations / self.total_evaluations) * 100
            if len(self.iteration_times) > 0:
                avg_time = sum(self.iteration_times) / len(self.iteration_times)
                remaining_iterations = self.total_evaluations - self.total_iterations
                eta_seconds = avg_time * remaining_iterations
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_str = f"ETA: {eta_hours}h {eta_minutes}m"
            else:
                eta_str = "ETA: calculating..."
            self.log_message(f"[OPTIMIZATION] Iteration {self.total_iterations}/{self.total_evaluations} ({progress_pct:.1f}%) | {eta_str}")
        else:
            self.log_message(f"[OPTIMIZATION] Iteration {self.total_iterations}")
        self.log_message(f"[OPTIMIZATION] Testing timesteps: {timesteps}")
        result = self.fitness_function(timesteps)
        iteration_time = time.time() - iteration_start
        self.iteration_times.append(iteration_time)
        if len(self.iteration_times) > 10:
            self.iteration_times = self.iteration_times[-10:]
        return result

    def optimize(self, num_timesteps, init_points=40, n_iter=200):
        self.total_evaluations = init_points + n_iter
        self.log_message("Starting Bayesian optimization for NeoDiff time schedule...")
        self.log_message(f"Configuration:")
        self.log_message(f"   - Number of timesteps: {num_timesteps}")
        self.log_message(f"   - Initial random points: {init_points}")
        self.log_message(f"   - Bayesian optimization iterations: {n_iter}")
        self.log_message(f"   - Total evaluations: {self.total_evaluations}")
        self.log_message(f"   - Model: {self.model_name}")
        self.log_message(f"   - Device: GPU {self.device}")
        self.log_message(f"   - Batch size: {self.batch_size}")
        self.log_message("="*80)
        pbounds = {f'x{i}': (0, 1) for i in range(num_timesteps)}
        optimizer = BayesianOptimization(
            f=self.target_function,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        self.log_message(f"Single-stage Bayesian optimization: init_points={init_points}, n_iter={n_iter}")
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        self.log_message("="*80)
        self.log_message("OPTIMIZATION COMPLETED!")
        self.log_message("="*80)
        self.log_message(f"Best BLEU score: {self.best_bleu:.4f}")
        self.log_message(f"Best timesteps: {self.best_timesteps}")
        self.log_message(f"Total evaluations completed: {self.total_iterations}")
        self.log_message("="*80)
        return optimizer.max

    def test_with_schedule(self, timesteps, steps=20):
        self.log_message(f"Testing with optimized schedule on TEST set: {timesteps}")
        prev_subset = self.subset
        self.subset = "test"
        try:
            test_results = self.evaluate_schedule(timesteps, steps=steps)
        finally:
            self.subset = prev_subset
        self.log_message(f"Test results with optimized schedule: {test_results}")
        return test_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for NeoDiff time schedule"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Name of the trained NeoDiff model to optimize")
    parser.add_argument("--timesteps", type=int, default=20, help="Number of timesteps to optimize for")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use for evaluation")
    parser.add_argument("--init-points", type=int, default=40, help="Number of initial random points for Bayesian optimization")
    parser.add_argument("--n-iter", type=int, default=200, help="Number of Bayesian optimization iterations")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate with a predefined schedule (no optimization)")
    parser.add_argument("--predefined-schedule", type=str, default=None, help="Comma-separated list of timesteps for evaluation (used with --eval-only)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for fairseq-generate evaluation")
    parser.add_argument("--discrete", action="store_true", help="Discretize timesteps to 0.1 granularity within [0,1] (round to 1 decimal)")
    parser.add_argument("--subset", type=str, default="valid", choices=["train", "valid", "test"], help="Data subset to run fairseq-generate on during optimization (default: valid)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    optimizer = NeoDiffScheduleOptimizer(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        discrete=args.discrete,
        subset=args.subset
    )
    if args.eval_only:
        if args.predefined_schedule:
            timesteps = [float(x) for x in args.predefined_schedule.split(',')]
        else:
            timesteps = [0.233, 0.869, 0.226, 0.639, 0.496, 0.449, 0.349, 0.205, 0.144, 0.601, 
                        0.574, 0.322, 0.063, 0.939, 0.951, 0.691, 0.123, 0.476, 0.090, 0.675]
            timesteps = timesteps[:args.timesteps]
        optimizer.test_with_schedule(timesteps)
    else:
        result = optimizer.optimize(
            num_timesteps=args.timesteps,
            init_points=args.init_points,
            n_iter=args.n_iter
        )
        print(f"\nOptimization Results:")
        print(f"Best BLEU: {result['target']:.4f}")
        print(f"Best parameters: {result['params']}")
        if optimizer.best_timesteps:
            optimizer.test_with_schedule(optimizer.best_timesteps)


if __name__ == "__main__":
    main() 