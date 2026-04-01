# @GonzaloMartinGarcia
# This file contains Marigold's exponential LR scheduler. 
# https://github.com/prs-eth/Marigold/blob/main/src/util/lr_scheduler.py

# Author: Bingxin Ke
# Last modified: 2024-02-22

import numpy as np
import math

class IterExponential:
    
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


class IterStep:
    
    def __init__(self, total_iter_length, warmup_steps=0, gamma=0.1, step_size=1000) -> None:
        """
        Customized iteration-wise step scheduler with warmup.
        Combines linear warmup with step decay learning rate scheduling.

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Minimum LR ratio (clamps the decay to not go below this)
            warmup_steps (int): Number of warmup steps for linear increase from 0 to 1
            gamma (float): Multiplicative factor for step decay (default: 0.1)
            step_size (int): Number of iterations between each decay step (default: 1000)
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        # self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.step_size = step_size

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            # Linear warmup from 0 to 1
            alpha = 1.0 * n_iter / self.warmup_steps
        else:
            # Step decay after warmup
            actual_iter = n_iter
            alpha = self.gamma ** (actual_iter // self.step_size)
        return alpha


class IterCosine:
    
    def __init__(self, total_iter_length, min_lr_ratio=1e-1, warmup_steps=100) -> None:
        """
        Customized iteration-wise cosine annealing scheduler with warmup.
        Implements cosine annealing with half-cycle cosine after warmup.

        Args:
            total_iter_length (int): Expected total iteration number
            min_lr_ratio (float): Minimum LR ratio (min_lr / base_lr)
            warmup_steps (int): Number of warmup steps for linear increase from 0 to 1
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            # Linear warmup from 0 to 1
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            # Clamp to minimum ratio at the end
            alpha = self.min_lr_ratio
        else:
            # Cosine annealing after warmup
            actual_iter = n_iter - self.warmup_steps
            alpha = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * actual_iter / self.effective_length)
            )
        return alpha