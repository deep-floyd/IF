# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import torch
import numpy as np


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights


class StaticSampler(ABC):

    def sample(self, batch_size, device, static_step=100):
        indices_np = np.ones(batch_size, dtype=np.int) * static_step
        weights_np = np.ones(batch_size, dtype=np.int)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
