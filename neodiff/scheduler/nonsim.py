import math

import torch
from scipy.stats import poisson

from .scheduler import Scheduler


class NonSimScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.trans_lambda = lambda x: -self.trans_alpha_bar(x).log() * (self.num_tau / 2)
        self.trans_lambda = lambda x: -self.trans_alpha_bar(x).log()
    
    def reverse(self, x_t, x_0_hat, ts, tau_hat, mask, noise=None, method=None, scores=None):
        # tau_hat = tau_hat - self.inference_step  # bsz
        # tau_hat = (tau_hat * self.num_tau).floor().clamp(0, self.max_tau) / self.max_tau
        
        if method == "normal":
            pass

        elif method == "cmlm":
            # lenght = mask.sum(-1, keepdim=True).cpu()  # bsz x 1
            # rank = torch.arange(mask.size(-1))
            # rank = ((rank + 0.5) / lenght).clamp(max=1)

            # taus = poisson.isf(rank, self.trans_lambda(t).cpu())
            # # taus = torch.from_numpy(taus).to(x_t).clamp(0, self.max_tau) / self.max_tau
            # taus = torch.from_numpy(taus).to(x_t).clamp(0, self.max_tau)

            # scores[~mask] = 1
            # sorted_index = scores.argsort(1)
            # tau_hat.scatter_(1, sorted_index, taus)
            tau_hat = self.create_labels(-scores, mask, ts).clamp(0, self.max_tau)

        else:
            raise NotImplementedError(method)

        # elif method == "det":
        #     tau_hat = self.get_next_t(ts)
        #     tau_hat = (t * self.num_tau).floor().clamp(0, self.max_tau) / self.max_tau

        x_t = self.forward(x_0_hat, tau_hat, noise)  # bsz x len x dim
        return x_t, self.get_next_t(ts), tau_hat


class DeterministicScheduler(NonSimScheduler):
    @property
    def pred_t(self):
        return False

    def sample_taus(self, ts, **kwargs):
        return ts
    

class BinomialScheduler(NonSimScheduler):
    def __init__(self, trans_schedule="linear", trans_args=dict(), *args, **kwargs):
        super().__init__(trans_schedule, trans_args, *args, **kwargs)

        self.num_t = trans_args["num_t"]
        # correction_factor = trans_args.correction_factor

        steps = torch.linspace(0, 1, self.num_t)
        alpha_bar = self.trans_alpha_bar(steps)
        alpha_bar_prev = torch.cat((torch.tensor([1.0]), alpha_bar[:-1]))
        alpha = alpha_bar / alpha_bar_prev
        alpha = alpha.pow(self.num_taus / 2)
        # beta = 1 - alpha

        # rescale and correct max_t+, max_tau- -> correction_factor+
        # alpha = 1 - beta / beta.sum() * self.max_tau * correction_factor

        self.p_bar = torch.zeros(self.num_t, self.num_taus)
        self.p_bar[0, 0], self.p_bar[0, 1] = alpha[0], 1 - alpha[0]

        p = torch.eye(self.num_taus)
        for t in range(1, self.num_t):
            torch.diagonal(p)[:-1].fill_(alpha[t])
            torch.diagonal(p, 1).fill_(1 - alpha[t])
            self.p_bar[t] = self.p_bar[t - 1] @ p

    @property
    def pred_t(self):
        return True
    
    def sample_taus(self, ts):
        p_bar = self.p_bar[ts]
        taus = torch.multinomial(p_bar, 1, True)  # bsz x len
        return taus / self.max_tau
    
    def to(self, dtype=None, device=None):
        super().to(dtype, device)
        self.p_bar.to(dtype=dtype, device=device)


class PoissonScheduler(NonSimScheduler):
    def __init__(
        self,
        num_taus=2,
        trans_schedule="linear",
        trans_args=dict(),
        noise_schedule="linear",
        noise_args=dict(),
    ):
        max_tau = num_taus - 1

        sigma_factor = 2.5
        sqrt_schedule_constant = (sigma_factor + math.sqrt(sigma_factor**2 + 4 * max_tau)) / 2
        trans_args["schedule_constant"] = sqrt_schedule_constant**2

        tau_constant = max(sqrt_schedule_constant / (sigma_factor * math.sqrt(2)) - 1, 0)
        self.tau_factor = lambda ts: tau_constant * (1 - (2 * ts - 1)**2) + 1

        super().__init__(num_taus, trans_schedule, trans_args, noise_schedule, noise_args)

        # self.trans_lambda = lambda x: sqrt_schedule_constant**2 * x

        if trans_schedule == "constant":
            self.trans_lambda = lambda x: 2 * max_tau * x
            self.tau_factor = lambda x: max_tau * x / 3
        
        elif trans_schedule == "constant_sqrt":
            self.trans_lambda = lambda x: 4 * max_tau * x
            self.tau_factor = lambda x: max_tau * x**0.5

        elif trans_schedule == "constant_cbrt":
            self.trans_lambda = lambda x: 3 * max_tau * x
            self.tau_factor = lambda x: 2 / 3 * max_tau * x**(1/3)

        elif trans_schedule == "constant_clamp":
            self.trans_lambda = lambda x: 2 * max_tau * x
            self.tau_factor = lambda x: torch.clamp(2 * x, 0, 1) * max_tau / 3

        elif trans_schedule == "constant_double":
            self.trans_lambda = lambda x: 2 * max_tau * x
            self.tau_factor = lambda x: 2 * x * max_tau / 3

        elif trans_schedule == "constant_triple":
            self.trans_lambda = lambda x: 2 * max_tau * x
            self.tau_factor = lambda x: x * max_tau

        elif trans_schedule == "constant_same_1":
            self.trans_lambda = lambda x:  max_tau * x
            self.tau_factor = lambda x: max_tau * x

        elif trans_schedule == "constant_same":
            self.trans_lambda = lambda x: 2 * max_tau * x
            self.tau_factor = lambda x: 2 * x * max_tau

        elif trans_schedule == "constant_same_triple":
            self.trans_lambda = lambda x: 3 * max_tau * x
            self.tau_factor = lambda x: 3 * x * max_tau

        elif trans_schedule == "constant_same_4":
            self.trans_lambda = lambda x: 4 * max_tau * x
            self.tau_factor = lambda x: 4 * max_tau * x

        # lambda_scale = trans_args["lambda_scale"]
        # self.trans_lambda = lambda x: -self.trans_alpha_bar(x).log() * (self.num_tau / 2)

        # max_lambda = self.trans_lambda(torch.tensor(1))
        # self.tau_scale = (max_lambda + 3 * max_lambda.sqrt()).item()

    @property
    def pred_t(self):
        return True
    
    def process_tau(self, ts, taus, trans_lambda, **kwargs):
        taus = (taus - trans_lambda) / (trans_lambda.sqrt() + 1e-8) * self.tau_factor(ts) + trans_lambda
        # return taus.clamp(0, self.max_tau) / self.max_tau
        return taus.round().clamp(0, self.max_tau)

    def sample_taus(self, ts, **kwargs):
        trans_lambda = self.trans_lambda(ts.float()).type_as(ts)  # bsz x len
        taus = torch.poisson(trans_lambda)
        taus = self.process_tau(ts, taus, trans_lambda, **kwargs)
        taus[ts > 0.99] = 1
        return taus
    
    def create_labels(self, loss, mask, ts, **kwargs):
        lenght = mask.sum(-1, keepdim=True)  # bsz x 1
        ref_rank = torch.arange(mask.size(-1), device=loss.device)
        ref_rank = ((ref_rank + 0.5) / lenght).clamp(0, 1)

        # all possible taus in the sentence
        trans_lambda = self.trans_lambda(ts.float()).type_as(ts)
        ref_taus = poisson.ppf(ref_rank.cpu(), trans_lambda.cpu())
        ref_taus = torch.from_numpy(ref_taus).to(loss.device)
        ref_taus = self.process_tau(ts, ref_taus, trans_lambda, **kwargs).long()

        loss = loss.clone()
        loss[~mask] = torch.inf
        loss_rank = loss.argsort(1).argsort(1)
        
        labels = torch.gather(ref_taus, 1, loss_rank)
        labels[~mask] = -100
        return labels
