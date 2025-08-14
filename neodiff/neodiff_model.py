from random import random

import torch
from torch import nn
from torch.nn import functional as F

from fairseq import utils
from fairseq.models import register_model, register_model_architecture, transformer
from fairseq.models.nat import NATransformerModel, cmlm_base_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .modules import NeoDiffEncoder, NeoDiffDecoder, TauPredictor
from .scheduler import *

@register_model("neodiff")
class NeoDiff(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--model-dim",
            type=int, metavar="N",
            help="The dimension of the model"
        )
        parser.add_argument(
            "--latent-dim",
            type=int, metavar="N",
            help="The dimension of $z_t$"
        )

        parser.add_argument(
            "--share-project-in-dim",
            action="store_true",
            help="Share projection layers of the encoder and decoder"
        )

        parser.add_argument(
            "--scheduler",
            type=str, metavar="STR", default="deterministic",
            help="The Scheduler used"
        )

        parser.add_argument(
            "--num-tau",
            type=int, metavar="N", default=10,
        )
        parser.add_argument(
            "--tau-type",
            type=str, metavar="STR", default="normal",
        )

        parser.add_argument(
            "--trans-schedule",
            type=str, metavar="STR", default="linear",
            help="The transition schedule"
        )
        parser.add_argument(
            "--num-t",
            type=int, metavar="N", default=1000,
        )
        # parser.add_argument(
        #     "--lambda-scale",
        #     type=float, metavar="D", default=500,
        # )

        parser.add_argument(
            "--noise-schedule",
            type=str, metavar="STR", default="linear",
        )
        parser.add_argument(
            "--forward-coeff-type",
            type=str, metavar="STR", default="sqrt",
        )
        parser.add_argument(
            "--reverse-coeff-type",
            type=str, metavar="STR", default="pred_epsilon",
        )
        parser.add_argument(
            "--eta",
            type=float, metavar="D", default=0.0,
        )
        parser.add_argument(
            "--var-type",
            type=str, metavar="STR", default="fixed_large",
        )

        parser.add_argument(
            "--noise-factor",
            type=float, metavar="D", default=1.0,
            help="The noise factor during training"
        )

        parser.add_argument(
            "--embed-norm",
            action="store_true",
            help="Add embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-affine",
            action="store_true",
            help="Add elementwise affine parameters to the embedding layer normalization"
        )
        parser.add_argument(
            "--embed-norm-before-proj",
            action="store_true",
            help="Put the embedding layer normalization before the projection layers"
        )

        parser.add_argument(
            "--self-cond",
            action="store_true",
            help="Self-conditioning"
        )
        parser.add_argument(
            "--self-cond-before-proj",
            action="store_true",
            help="Concatenate self-conditioning embeddings before the projection layers"
        )

        parser.add_argument(
            "--pred-zt",
            action="store_true",
        )
        parser.add_argument(
            "--new-noise",
            action="store_true",
        )
        parser.add_argument(
            "--new-ts",
            action="store_true",
        )

        parser.add_argument(
            "--rounding-loss",
            action="store_true",
            help="Use the rounding loss instead of the anchor loss"
        )

        parser.add_argument(
            "--tau-predictor-layers",
            type=int, metavar="N", default=2,
            help=""
        )
        parser.add_argument(
            "--pred-tau-wo-t",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--pred-tau-detach",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--pred-tau-detach-all",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--tau-loss-anchor",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--tau-mse",
            action="store_true",
            help=""
        )

        parser.add_argument(
            "--mse-loss-weight",
            type=str, metavar="STR", default="none",
        )
        parser.add_argument(
            "--anchor-loss-weight",
            type=str, metavar="STR", default="none",
        )

        # parser.add_argument(
        #     "--store-tau-predictions",
        #     action="store_true",
        #     help=""
        # )
        # parser.add_argument(
        #     "--store-tau-predictions-path",
        #     type=str, metavar="STR", default=None,
        #     help=""
        # )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        if args.scheduler == "det":
            self.scheduler = DeterministicScheduler(
                2,
                args.trans_schedule, {"schedule_constant": 1.0},
                args.noise_schedule, {"forward_coeff_type": args.forward_coeff_type},
            )
        
        elif args.scheduler == "binomial":
            self.scheduler = BinomialScheduler(
                args.num_tau,
                args.trans_schedule, {"num_t": args.num_t},
                args.noise_schedule, {"forward_coeff_type": args.forward_coeff_type},
            )

        elif args.scheduler == "poisson":
            self.scheduler = PoissonScheduler(
                # args.trans_schedule, {"lambda_scale": args.lambda_scale},
                args.num_tau,
                args.trans_schedule, {},
                args.noise_schedule, {"forward_coeff_type": args.forward_coeff_type},
            )
        
        else:
            self.scheduler = SimpleScheduler(
                args.num_tau,
                args.trans_schedule, {},
                args.noise_schedule, {
                    "forward_coeff_type": args.forward_coeff_type,
                    "reverse_coeff_type": args.reverse_coeff_type,
                    "eta": args.eta,
                    "var_type": args.var_type
                },
            )

        self.tau_predictor = TauPredictor(args, self.decoder.project_in_dim, self.decoder.embed_positions, self.decoder.embed_time)
        self.tau_criterion = nn.CrossEntropyLoss()

        self.tau_predictions = []
        # self.store_tau_predictions = args.store_tau_predictions
        # self.store_tau_predictions_path = args.store_tau_predictions_path

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, project_in_dim):
        return NeoDiffEncoder(args, src_dict, embed_tokens, project_in_dim)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, project_in_dim, project_out_dim):
        decoder = NeoDiffDecoder(args, tgt_dict, embed_tokens, project_in_dim, project_out_dim)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """ Build a new model instance """

        transformer.base_architecture(args)
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = transformer.DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        latent_dim = args.latent_dim
        model_dim = args.model_dim

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, latent_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, latent_dim, args.encoder_embed_path
            )

        # projection layers
        if latent_dim != model_dim:
            encoder_project_in_dim = nn.Linear(latent_dim, model_dim, bias=False)
            decoder_project_in_dim = (
                encoder_project_in_dim if args.share_project_in_dim
                else nn.Linear(latent_dim, model_dim, bias=False)
            )
            
            decoder_project_out_dim = nn.Linear(model_dim, latent_dim, bias=False)
        
        else:
            encoder_project_in_dim = nn.Identity()
            decoder_project_in_dim = nn.Identity()
            decoder_project_out_dim = nn.Identity()

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, encoder_project_in_dim)
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens,
            decoder_project_in_dim, decoder_project_out_dim
        )

        return cls(args, encoder, decoder)
    
    def compute_loss(self, logits, tgt_tokens, mask, label_smoothing=0.0, loss_weight=1.0):
        logp = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(
            logp.transpose(-1, -2),
            tgt_tokens,
            reduction="none"
        )

        if label_smoothing > 0:
            loss = (
                nll_loss * (1 - label_smoothing) - logp.mean(-1) * label_smoothing
            )
        else:
            loss = nll_loss

        loss = loss * loss_weight
        return loss[mask].mean()

    def forward(self, src_tokens, src_lengths, _, tgt_tokens):
        """ Compute training losses """
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        # print everything about encoder_out, but not contents
        # print(encoder_out.keys())
        # print(type(encoder_out))
        # print(len(encoder_out))
        # for key, value in encoder_out.items():
            # print(f"  {key}: {value}")
        # exit()

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        mask = tgt_tokens.ne(self.pad)

        # diffusion
        x_0 = self.decoder.forward_embedding(tgt_tokens)

        sample_indep = self.args.tau_type == "indep" or self.args.tau_type == "uniform"
        ts, taus = self.scheduler.sample_timesteps(tgt_tokens.size(), sample_indep)
        rescaled_taus = taus / self.args.max_tau

        if self.args.tau_type == "uniform":
            taus = (ts * self.args.num_tau).floor().clamp_(max=self.args.num_tau - 1) / (self.args.num_tau - 1)

        # tokens = tgt_tokens.clone()
        # tokens[taus == 1] = self.unk
        # x_t = self.decoder.forward_embedding(tokens)
            
        # random replace
        # rep_mask = torch.rand_like(taus) < ts[:, None] / 2
        # rand_tokens = torch.randint_like(tgt_tokens, len(self.decoder.embed_tokens.weight))
        # x_t[rep_mask] = self.decoder.forward_embedding(rand_tokens[rep_mask])

        noise = torch.randn_like(x_0) * self.args.noise_factor
        # noise = self.decoder.forward_embedding(torch.full_like(tgt_tokens, self.unk))

        x_t = self.scheduler.forward(x_0, taus, noise)

        # self-conditioning
        prev_x_0_hat = torch.zeros_like(x_0)
        if self.args.self_cond and random() < 0.5:
            with torch.no_grad():
                # print(type(self.decoder(x_t, rescaled_taus, mask, encoder_out, prev_x_0_hat)))
                # exit()
                prev_x_0_hat = self.decoder(x_t, rescaled_taus, mask, encoder_out, prev_x_0_hat)[0]

        if self.args.pred_zt and random() < 0.01:
            with torch.no_grad():
                x_0_hat = self.decoder(x_t, rescaled_taus, mask, encoder_out, prev_x_0_hat)[0]

                if self.args.new_noise:
                    noise = torch.randn_like(x_0) * self.args.noise_factor

                if self.args.new_ts:
                    ts, taus = self.scheduler.sample_timesteps(tgt_tokens.size(), sample_indep)

                x_t = self.scheduler.forward(x_0_hat, taus, noise)
        
        x_0_hat = self.decoder(x_t, rescaled_taus, mask, encoder_out, prev_x_0_hat)[0]
        logits = self.decoder.output_layer(x_0 if self.args.rounding_loss else x_0_hat)

        # taus_mean = taus.mean(-1, keepdim=True)
        # taus_mean[taus_mean == 0] = 1
        # loss_factor = taus / taus_mean

        diffusion_loss = (x_0_hat - x_0).square().mean(-1)

        anchor_loss = F.cross_entropy(
            logits.transpose(-1, -2),
            tgt_tokens,
            reduction="none"
        )

        if self.args.mse_loss_weight == "linear":
            mse_loss_weight = ts + 1
        elif self.args.mse_loss_weight == "neg_linear":
            mse_loss_weight = 2 - ts
        elif self.args.mse_loss_weight == "quadratic":
            mse_loss_weight = ts * (1 - ts) + 1
        else:
            mse_loss_weight = 1.0

        if self.args.anchor_loss_weight == "linear":
            anchor_loss_weight = ts + 1
        elif self.args.anchor_loss_weight == "neg_linear":
            anchor_loss_weight = 2 - ts
        elif self.args.anchor_loss_weight == "quadratic":
            anchor_loss_weight = ts * (1 - ts) + 1
        else:
            anchor_loss_weight = 1.0
            

        losses = {
            "diffusion": {
                "loss": (mse_loss_weight * diffusion_loss)[mask].mean()
            },

            "anchor": {
                "loss": self.compute_loss(
                    logits,
                    tgt_tokens,
                    mask,
                    self.args.label_smoothing,
                    anchor_loss_weight,
                )
            },
            
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

        if self.scheduler.pred_t:
            # tau_hat = self.tau_predictor(x_0_hat.detach(), mask, encoder_out)
            tau_logits, tau_pred = self.tau_predictor(
                x_0_hat.detach() if self.args.pred_tau_detach else x_0_hat,
                mask,
                None if self.args.pred_tau_wo_t else ts,
                encoder_out
            )


            tau_label_loss = diffusion_loss
            if self.args.tau_loss_anchor:
                tau_label_loss += anchor_loss

            tau_labels = self.scheduler.create_labels(tau_label_loss, mask, ts)
            tau_loss = self.tau_criterion(tau_logits.view(-1, self.args.num_tau), tau_labels.view(-1))

            if self.args.tau_mse:
                losses["tau_mse"] = {
                    "loss": (
                        tau_pred / self.args.max_tau
                        - tau_labels / self.args.max_tau
                    )[mask].square().mean()
                }

            else:
                losses["tau_mse"] = {
                    "loss": (
                        tau_logits.argmax(-1) / self.args.max_tau
                        - tau_labels / self.args.max_tau
                    ).detach()[mask].square().mean()
                }

                losses["tau"] = {"loss": tau_loss}

        return losses

    def forward_decoder(self, x_t, t, taus, mask, encoder_out, prev_x_0_hat=None, custom_timesteps=None,current_step = 0):
        """ Sample z_{t-1} given z_t """
        # predict z_0
        rescaled_taus = taus / self.args.max_tau
        x_0_hat = self.decoder(x_t, rescaled_taus, mask, encoder_out, prev_x_0_hat)[0]

        input_t = self.scheduler.get_next_t(t,custom_timesteps,current_step)

        if self.scheduler.pred_t and not self.args.decoding_wo_pred_t:
            tau_logits, tau_pred = self.tau_predictor(
                x_0_hat,
                mask,
                None if self.args.pred_tau_wo_t else input_t,
                encoder_out
            )

            if self.args.tau_mse:
                tau_hat = tau_pred.clamp(0, self.args.max_tau)
            else:
                tau_hat = tau_logits.argmax(-1)

        else:
            tau_hat = self.scheduler.sample_taus(input_t)

        # clamping trick
        if self.args.clamping:
            tokens = self.decoder.output_layer(x_0_hat).argmax(-1)
            x_0_hat = self.decoder.forward_embedding(tokens)

        # sample z_{t-1}
        noise = torch.randn_like(x_t) * self.args.decoding_nf
        x_t, t, tau_hat = self.scheduler.reverse(x_t, x_0_hat, t, tau_hat, mask, noise, method=self.args.decoding_method, scores=self.forward_output_layer(x_0_hat, mask)[1])
        return x_t, t, tau_hat, x_0_hat

    def forward_output_layer(self, x_t, mask):
        scores, tokens = self.decoder.output_layer(x_t).log_softmax(-1).max(-1)
        return tokens, scores, mask

    def initialize_z_t(self, encoder_out):
        """ Sample z_T """
        # length prediction
        pred_length = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = pred_length.clamp_(min=2).max()
        # noise = self.decoder.forward_embedding(torch.full_like(pred_length, 3))[:, None].expand(-1, max_length, -1)
        # z_t, taus = self.scheduler.get_init_samples(len(pred_length), max_length, self.args.latent_dim, samples=noise)
        z_t, t, taus = self.scheduler.get_init_samples(len(pred_length), max_length, self.args.latent_dim)

        return z_t, t, taus, pred_length

    def regenerate_beam(self, pred_length, lenght_beam_size, noise_beam_size):
        pred_length = (
            pred_length[:, None, None]
            + utils.new_arange(pred_length, 1, noise_beam_size, lenght_beam_size).transpose(-1, -2)
            - lenght_beam_size // 2
        ).flatten().to(pred_length)  # (bsz * lenght_beam_size * noise_beam_size)

        max_length = pred_length.clamp_(min=2).max()
        # noise = self.decoder.forward_embedding(torch.full_like(pred_length, 3))[:, None].expand(-1, max_length, -1)
        # z_t, taus = self.scheduler.get_init_samples(len(pred_length), max_length, self.args.latent_dim, samples=noise)
        z_t, t, taus = self.scheduler.get_init_samples(len(pred_length), max_length, self.args.latent_dim)

        return z_t, t, taus, pred_length
    
    def to(self, device=None,dtype=None):
        # print("[DBEUG in difformer.py] to function, dtype is ", dtype, "device is ", device)
        super().to(device=device, dtype=dtype)

        dummy_param = next(self.parameters())
        self.scheduler.to(dtype=dummy_param.dtype, device=dummy_param.device)
        return self
    
    def cuda(self):
        super().cuda()

        dummy_param = next(self.parameters())
        self.scheduler.to(dtype=dummy_param.dtype, device=dummy_param.device)
        return self

    def save_tau_predictions(self, output_dir):
        # if self.store_tau_predictions and self.tau_predictions:
            # exit()
            json_path = os.path.join(output_dir, "tau_predictions.json")
            
            # Load existing predictions if the file already exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_predictions = json.load(f)
            else:
                existing_predictions = []

            # Append new predictions to the existing ones
            existing_predictions.extend(self.tau_predictions)
            
            # Save the updated predictions to the JSON file
            with open(json_path, 'w') as f:
                json.dump(existing_predictions, f, indent=4)
            
            # Clear predictions after saving to avoid duplication
            self.tau_predictions = [] 


@register_model_architecture("neodiff", "neodiff")
def base_architecture(args):
    args.model_dim = getattr(args, "model_dim", 512)
    args.latent_dim = getattr(args, "latent_dim", 128)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = args.model_dim
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = args.model_dim
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )

    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.latent_dim)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_input_dim)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_project_in_dim = getattr(args, "share_project_in_dim", False)

    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)

    args.scheduler = getattr(args, "scheduler", "deterministic")
    args.num_tau = getattr(args, "num_tau", 10)
    args.max_tau = args.num_tau - 1
    args.tau_type = getattr(args, "tau_type", "normal")

    args.trans_schedule = getattr(args, "trans_schedule", "linear")
    args.num_t = getattr(args, "num_t", 1000)
    # args.lambda_scale = getattr(args, "lambda_scale", 500)
    # args.num_tau = getattr(args, "lambda_scale", 500)

    args.noise_schedule = getattr(args, "noise_schedule", "linear")
    args.forward_coeff_type = getattr(args, "forward_coeff_type", "sqrt")
    args.reverse_coeff_type = getattr(args, "reverse_coeff_type", "pred_epsilon")
    args.eta = getattr(args, "eta", 0.0)
    args.var_type = getattr(args, "var_type", "fixed_large")

    args.tau_predictor_layers = getattr(args, "tau_predictor_layers", 1)

    args.noise_factor = getattr(args, "noise_factor", 1.0)
    args.rescale_factor = getattr(args, "rescale_factor", 1.0)

    args.embed_norm = getattr(args, "embed_norm", False)
    args.embed_norm_affine = getattr(args, "embed_norm_affine", False)
    args.embed_norm_before_proj = getattr(args, "embed_norm_before_proj", False)

    args.self_cond = getattr(args, "self_cond", False)
    args.self_cond_before_proj = getattr(args, "self_cond_before_proj", False)

    args.pred_zt = getattr(args, "pred_zt", False)
    args.new_noise = getattr(args, "new_noise", False)
    args.new_ts = getattr(args, "new_ts", False)

    args.rounding_loss = getattr(args, "rounding_loss", False)

    args.pred_tau_wo_t = getattr(args, "pred_tau_wo_t", False)
    args.pred_tau_detach = getattr(args, "pred_tau_detach", False)
    args.pred_tau_detach_all = getattr(args, "pred_tau_detach_all", False)
    args.tau_loss_anchor = getattr(args, "tau_loss_anchor", False)
    args.tau_mse = getattr(args, "tau_mse", False)

    args.mse_loss_weight = getattr(args, "mse_loss_weight", "none")
    args.anchor_loss_weight = getattr(args, "anchor_loss_weight", "none")

@register_model_architecture("neodiff", "neodiff_base")
def neodiff_base(args):
    args.model_dim = getattr(args, "model_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    base_architecture(args)


@register_model_architecture("neodiff", "neodiff_iwslt_de_en")
def neodiff_iwslt_de_en(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    base_architecture(args)

@register_model_architecture("transformer", "transformer_base")
def transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    transformer.base_architecture(args)

@register_model_architecture("cmlm_transformer", "cmlm_transformer_iwslt")
def cmlm_transformer_base(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    cmlm_base_architecture(args)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_base")
def cmlm_transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    cmlm_base_architecture(args)
