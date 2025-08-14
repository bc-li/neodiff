from abc import ABC, abstractmethod

import torch


def get_noise_schedule(schedule_name, **kwargs):
    if schedule_name == "linear":
        min_beta = 0.1
        max_beta = 20
        alpha_bar = lambda x: torch.exp(-(max_beta - min_beta) / 2 * x ** 2 - min_beta * x)

    elif schedule_name == "cosine":
        shift = 0.008
        alpha_bar = lambda x: torch.clamp(torch.cos((x + shift) / (1 + shift) * torch.pi / 2) ** 2, 1e-8)

    elif schedule_name == "sqrt":
        shift = 0.0001
        alpha_bar = lambda x: torch.clamp(1 - torch.sqrt(x + shift), 1e-8)

    elif schedule_name == "edm":
        rho = 7
        min_sigma = 0.002 ** (1 / rho)
        max_sigma = 80 ** (1 / rho)
        alpha_bar = lambda x: 1 / ((max_sigma + (1 - x) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    elif schedule_name == "cdcd":
        rho = 7
        min_sigma = 1 ** (1 / rho)
        max_sigma = 300 ** (1 / rho)
        alpha_bar = lambda x: 1 / ((max_sigma + (1 - x) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    elif schedule_name.startswith("constant"):
        constant = kwargs["schedule_constant"]
        alpha_bar = lambda x: torch.exp(-constant * x)

    else:
        raise NotImplementedError(f"unknown noise schedule: {schedule_name}")
    
    return alpha_bar


class Scheduler(ABC):
    @abstractmethod
    def __init__(
        self,
        num_taus=2,
        trans_schedule="linear",
        trans_args=dict(),
        noise_schedule="linear",
        noise_args=dict(),
    ):
        
        self.num_taus = num_taus
        self.max_tau = num_taus - 1

        self.trans_alpha_bar = get_noise_schedule(trans_schedule, **trans_args)
        self.trans_args = trans_args

        self.noise_alpha_bar = get_noise_schedule(noise_schedule, **noise_args)
        self.noise_args = noise_args

        self.dtype = None
        self.device = None

        self.num_inference_steps = None
        self.inference_step = None
    
    @property
    @abstractmethod
    def pred_t(self):
        pass
    
    @abstractmethod
    def sample_taus(self, ts, **kwargs):
        pass

    def sample_timesteps(self, size, independent=False, **kwargs):
        if independent:
            ts = torch.rand(size, dtype=self.dtype, device=self.device)
        else:
            ts = torch.rand(size[:-1], dtype=self.dtype, device=self.device)[..., None].expand(size)
        
        taus = self.sample_taus(ts, **kwargs)
        return ts, taus
    
    def forward(self, x_0, taus, noise=None):
        alpha_bar = self.noise_alpha_bar(taus / self.max_tau)[..., None].type_as(x_0)  # bsz x len x 1
        beta_bar = 1 - alpha_bar

        noise = noise if noise is not None else torch.randn_like(x_0)
        if self.noise_args["forward_coeff_type"] == "sqrt":
            x_t = alpha_bar.sqrt() * x_0 + beta_bar.sqrt() * noise  # bsz x len x dim
        else:
            x_t = alpha_bar * x_0 + beta_bar * noise
        
        return x_t

    def init_reverse_args(self, num_inference_steps=1000):
        self.num_inference_steps = num_inference_steps
        self.inference_step = 1 / num_inference_steps
    
    def get_init_samples(self, *size, x_t=None, t=None, taus=None):
        x_t = x_t if x_t is not None else torch.randn(size, dtype=self.dtype, device=self.device)
        # taus = taus if taus is not None else torch.full(size[:-1], 1, dtype=self.dtype, device=self.device)
        taus = taus if taus is not None else torch.full(size[:-1], self.max_tau, dtype=self.dtype, device=self.device)
        t = t if t is not None else torch.ones(1, dtype=self.dtype, device=self.device)
        return x_t, t, taus
    
    def get_next_t(self, t,custom_timesteps = None, current_step = None):
        if custom_timesteps is None:
            return t - self.inference_step
        else:
            custom_timesteps_list = custom_timesteps.split(",")
            custom_timesteps_list = [float(timestep)  for timestep in custom_timesteps_list]
            timesteps = torch.tensor(custom_timesteps_list).to(t.device)
            sorted_timesteps, indices = torch.sort(timesteps, descending=False)
            # print("timesteps",timesteps)
            # exit()
            return sorted_timesteps[current_step]

    @abstractmethod
    def reverse(self, x_t, model_output, t, tau_hat, mask, noise=None):
        pass

    def to(self, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        return self
