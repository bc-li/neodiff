import torch
from .scheduler import Scheduler


class SimpleScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coeff_type = self.noise_args.pop("reverse_coeff_type")
        self.coeff_args = self.noise_args

        self.pred_coeff = None
        self.sample_coeff = None
        self.std = None

    @property
    def pred_t(self):
        return False

    def sample_taus(self, ts, **kwargs):
        return ts * self.max_tau
    
    def init_reverse_args(self, num_inference_steps=1000):
        super().init_reverse_args(num_inference_steps)

        steps = torch.linspace(0, 1, num_inference_steps, dtype=self.dtype, device=self.device)

        alpha_bar = self.noise_alpha_bar(steps)
        alpha_bar_prev = torch.cat((torch.tensor([1.0]), alpha_bar[:-1]))
        alpha = alpha_bar / alpha_bar_prev

        beta_bar = 1 - alpha_bar
        beta_bar_prev = 1 - alpha_bar_prev
        beta = 1 - alpha

        if self.coeff_type == "pred_score":
            self.sample_coeff = 1 + 0.5 * beta
            self.pred_coeff = beta

        elif self.coeff_type == "pred_epsilon":
            self.sample_coeff = 1 / alpha.sqrt()
            self.pred_coeff = -beta / (alpha * beta_bar).sqrt()

        elif self.coeff_type == "pred_start":
            self.sample_coeff = alpha.sqrt() * beta_bar_prev / beta_bar
            self.pred_coeff = alpha_bar_prev.sqrt() * beta / beta_bar

        elif self.coeff_type == "ddim_pred_epsilon":
            eta = torch.full_like(steps, self.coeff_args["eta"])
            self.sample_coeff = 1 / alpha.sqrt()
            self.pred_coeff = (beta_bar_prev - eta.square()).sqrt() - (beta_bar / alpha).sqrt()

        elif self.coeff_type == "ddim_pred_start":
            eta = torch.full_like(steps, self.coeff_args["eta"])
            self.sample_coeff = ((beta_bar_prev - eta.square()) / beta_bar).sqrt()
            self.pred_coeff = alpha_bar_prev.sqrt() - self.sample_coeff * alpha_bar.sqrt()

        if self.self.coeff_type.startswith("ddim"):
            self.std = eta
        
        elif self.coeff_args["var_type"] == "fixed_large":
            self.std = beta.sqrt()
        else:  # fixed_small
            self.std = torch.clamp(beta * beta_bar_prev / beta_bar, min=1e-20).sqrt()

    def reverse(self, x_t, model_output, t, tau_hat, mask, noise=None):
        sample_coeff = self.sample_coeff[t]
        pred_coeff = self.pred_coeff[t]
        std = self.std[t]

        noise = noise if noise is not None else torch.randn_like(x_t)
        x_t = sample_coeff * x_t + pred_coeff * model_output + std * noise
        return x_t, self.get_next_t(t), None
