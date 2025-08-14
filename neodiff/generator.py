import torch

from fairseq import utils
from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

class NeoDiffGenerator(IterativeRefinementGenerator):
    def __init__(
        self,
        tgt_dict,
        steps=20,
        early_stopping=0,
        length_beam_size=1,
        noise_beam_size=1,
        ppl_mbr=False,
        bleu_mbr=False,
        retain_history=False,
        retain_z_0_hat=False,
    ):
        super().__init__(
            tgt_dict=tgt_dict,
            retain_history=retain_history,
        )

        self.tgt_dict = tgt_dict

        self.steps = steps
        self.early_stopping = early_stopping

        self.length_beam_size = length_beam_size
        self.noise_beam_size = noise_beam_size
        self.beam_size = length_beam_size * noise_beam_size

        self.ppl_mbr = ppl_mbr
        self.bleu_mbr = bleu_mbr

        self.retain_z_0_hat = retain_z_0_hat

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model, reranker = models[0], None
        if self.ppl_mbr and len(models) > 1 and self.beam_size > 1:
            reranker = models[-1]

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])
        x_t, t, taus, pred_length = model.initialize_z_t(encoder_out)
        model.scheduler.init_reverse_args(self.steps)

        if self.beam_size > 1:
            # regenerate data
            beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, beam_order
            )
            x_t, t, taus, pred_length = model.regenerate_beam(
                pred_length, self.length_beam_size, self.noise_beam_size
            )

        mask = utils.new_arange(x_t, *x_t.size()[:-1]) < pred_length[:, None]
        prev_x_0_hat = torch.zeros_like(x_t)

        if self.retain_history:
            history = [model.forward_output_layer(x_t, mask)]

        # start decoding
        for _ in list(range(self.early_stopping, self.steps))[::-1]:
            x_t, t, taus, prev_x_0_hat = model.forward_decoder(
                x_t, t, taus, mask, encoder_out, prev_x_0_hat
            )

            if self.retain_history:
                output = prev_x_0_hat if self.retain_z_0_hat else x_t
                history.append(model.forward_output_layer(output, mask))
        
        # decoding finished
        finalized = model.forward_output_layer(prev_x_0_hat, mask)  # (tokens, scores, mask)

        def finalized_hypos(step, tokens, scores, mask):
            tokens = tokens[mask]
            scores = scores[mask]
            score = scores.mean()

            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": None,
                "alignment": None,
            }

        # (bsz * beam_size) x 1
        finalized = [[finalized_hypos(self.steps, t, s, m)] for t, s, m in zip(*finalized)]
        if self.retain_history:
            for i, f in enumerate(finalized):
                f[0]["history"] = [
                    finalized_hypos(step, t[i], s[i], m[i])
                    for step, (t, s, m) in enumerate(history)
                ]

        # select the best sentences
        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )
            
            elif self.bleu_mbr:
                finalized = self.bleu_rerank(finalized, self.beam_size)

            # aggregate information from beam
            scores = torch.stack([f[0]["score"] for f in finalized]).view(-1, self.beam_size)
            selected_idx = scores.argmax(-1)
            finalized = [finalized[idx + i * self.beam_size] for i, idx in enumerate(selected_idx)]

        return finalized
    
    def bleu_rerank(self, finalized, beam_size):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            return s

        for idx in range(0, len(finalized), beam_size):
            beam = [decode(f[0]["tokens"]) for f in finalized[idx:idx + beam_size]]
            for i, si in enumerate(beam):
                if not si:
                    continue

                finalized[idx + i][0]["score"].fill_(sum([
                    torch.as_tensor(sacrebleu.sentence_bleu(sj, [si]).score)
                    for j, sj in enumerate(beam)
                    if j != i and sj
                ]) / (beam_size - 1))

        return finalized





class NeoDiffTuningTGenerator(NeoDiffGenerator):
    '''
    Add global T tuning to NeoDiff
    '''
    # def __init__(self, *args, **kwargs):
    def __init__(
        self,
        tgt_dict,
        steps=20,
        early_stopping=0,
        length_beam_size=1,
        noise_beam_size=1,
        ppl_mbr=False,
        bleu_mbr=False,
        retain_history=False,
        retain_z_0_hat=False,
        **kwargs
    ):
        super().__init__(
            tgt_dict=tgt_dict,
            retain_history=retain_history,
        )

        self.tgt_dict = tgt_dict

        self.steps = steps
        self.early_stopping = early_stopping

        self.length_beam_size = length_beam_size
        self.noise_beam_size = noise_beam_size
        self.beam_size = length_beam_size * noise_beam_size

        self.ppl_mbr = ppl_mbr
        self.bleu_mbr = bleu_mbr

        self.retain_z_0_hat = retain_z_0_hat
        # self.tau_inspect = kwargs.get("tau_inspect", False)
        # self.tau_store_path = kwargs.get("tau_store_path", None)
        self.custom_timesteps = kwargs.get("custom_timesteps", None)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):    
       
        model, reranker = models[0], None 
        # here model indicates the neodiff model, reranker indicates the reranker model
        if self.ppl_mbr and len(models) > 1 and self.beam_size > 1:
            reranker = models[-1]

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])
        x_t, t, taus, pred_length = model.initialize_z_t(encoder_out)
        model.scheduler.init_reverse_args(self.steps)

        # Storage for tau and corresponding samples
        tau_token_records = []

        if self.beam_size > 1:
            # regenerate data
            beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, beam_order
            )
            x_t, t, taus, pred_length = model.regenerate_beam(
                pred_length, self.length_beam_size, self.noise_beam_size
            )

        mask = utils.new_arange(x_t, *x_t.size()[:-1]) < pred_length[:, None]
        prev_x_0_hat = torch.zeros_like(x_t)

        if self.retain_history:
            history = [model.forward_output_layer(x_t, mask)]

        # start decoding
        for step in list(range(self.early_stopping, self.steps))[::-1]:
            # step: current_step
            x_t, t, taus, prev_x_0_hat = model.forward_decoder(
                x_t, t, taus, mask, encoder_out, prev_x_0_hat,self.custom_timesteps,step
            )
            tau_token_records.append({
                "step": step,
                "tau": taus.tolist(),
                "src_tokens": src_tokens.tolist(),
                # "target_tokens": x_t.tolist(),
                "src_lengths": src_lengths.tolist(),
                "global_t": t.tolist()
            })

            if self.retain_history:
                output = prev_x_0_hat if self.retain_z_0_hat else x_t
                history.append(model.forward_output_layer(output, mask))
        # save_tau_predictions(self,tau_sample_records, self.tau_store_path)
        # decoding finished
        finalized = model.forward_output_layer(prev_x_0_hat, mask)  # (tokens, scores, mask)
 
        def finalized_hypos(step, tokens, scores, mask):
            tokens = tokens[mask]
            scores = scores[mask]
            score = scores.mean()

            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": None,
                "alignment": None,
            }

        # (bsz * beam_size) x 1
        finalized = [[finalized_hypos(self.steps, t, s, m)] for t, s, m in zip(*finalized)]
        if self.retain_history:
            for i, f in enumerate(finalized):
                f[0]["history"] = [
                    finalized_hypos(step, t[i], s[i], m[i])
                    for step, (t, s, m) in enumerate(history)
                ]

        # select the best sentences
        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )
            
            elif self.bleu_mbr:
                finalized = self.bleu_rerank(finalized, self.beam_size)

            # aggregate information from beam
            scores = torch.stack([f[0]["score"] for f in finalized]).view(-1, self.beam_size)
            selected_idx = scores.argmax(-1)
            finalized = [finalized[idx + i * self.beam_size] for i, idx in enumerate(selected_idx)]
        
        def save_tau_token_records(self, tau_token_records, output_dir):
            import os, json,time
            # current_time_formulated = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

            json_path = os.path.join(output_dir, "tau_token_records.json")
            # json_path = json_path.replace("tau_token_records.json", f"tau_token_records_{current_time_formulated}.json")
            # Load existing records if the file already exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        existing_records = json.load(f)
                    except json.JSONDecodeError:
                        existing_records = []  # If the file is empty or invalid, start with an empty list
            else:
                existing_records = []

            # Append new records to the existing ones
            existing_records.extend(tau_token_records)



            # Process finalized and tau_token_records
            for i in range(len(finalized)):
                # Get the decoding history from finalized
                decoding_history = finalized[i][0]['history']

                # Ensure tau_token_records is long enough (excluding the first initialization step in history)
                while len(tau_token_records) < len(decoding_history) - 1:
                    tau_token_records.append({})

                # Add token sequence to tau_token_records for each step (excluding the first initialization step in history)
                for j in range(1, len(decoding_history)):
                    token_sequence = decoding_history[j]['tokens'].tolist()
                    decoded_words = self.tgt_dict.string(token_sequence, bpe_symbol="@@ ")
                    tau_token_records[j - 1]['decoded_tokens'] = decoded_words
            # Save the updated records to the JSON file
            with open(json_path, 'w') as f:
                json.dump(existing_records, f, indent=4)
        # Save tau and token records
        # save_tau_token_records(self, tau_token_records, self.tau_store_path)
        return finalized
    