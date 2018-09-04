import torch
from typing import Tuple, Optional

import numpy as np
from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("span_start")
class SpanStartMetrics(Metric):
    """
    This :class:`Metric` takes the predicted span starts computed by a model, along with the gold
    span starts from the data, and computes exact match, accuracy, F1_m and F1_a scores.

    Note: the accuracy score indicates the success rate of predicting the correct span starts.
    """
    def __init__(self) -> None:
        self._total_em = 0.0

        # Used for computing F1 score per question (F1_m).
        self._total_precision = 0.0
        self._total_recall = 0.0

        # Used for computing F1 score per dataset (F1_a).
        self._total_correct_predicted_span_starts = 0.0
        self._total_predicted_span_starts = 0.0
        self._total_gold_span_starts = 0.0

        self._count = 0

    @overrides
    def __call__(self, predicted_span_starts: torch.Tensor,
                 gold_span_starts: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predicted_span_starts : ``torch.Tensor``
            A tensor of predicted span starts of shape (batch_size, 4).
        gold_span_starts : ``torch.Tensor``
            A tensor of gold span starts of shape (batch_size, d) where d <=4.
        """
        # Some sanity checks.
        if predicted_span_starts.dim() != gold_span_starts.dim():
            raise ConfigurationError("predicted_span_starts must have the same dimensions as gold_span_starts but "
                                     "found predicted_span_starts of dimension {} and gold_span_starts of dimension: {}"
                                     .format(predicted_span_starts.dim(), gold_span_starts.dim()))
        if predicted_span_starts.size(0) != gold_span_starts.size(0):
            raise ConfigurationError("predicted_span_starts and gold_span_starts must agree on the first dimension but "
                                     "found predicted_span_starts of shape {} and gold_span_starts of shape: {}"
                                     .format(predicted_span_starts.shape, gold_span_starts.shape))
        if predicted_span_starts.size(1) != gold_span_starts.size(1):
            padded_gold_span_starts = torch.full(predicted_span_starts.shape, -1).cuda()
            # padded_gold_span_starts = torch.full(predicted_span_starts.shape, -1)
            padded_gold_span_starts[:, :gold_span_starts.size(1)] = gold_span_starts
            gold_span_starts = padded_gold_span_starts

        # Sort both arrays for the three metrics.
        predicted_span_starts = np.sort(predicted_span_starts.detach().cpu().numpy())
        gold_span_starts = np.sort(gold_span_starts.detach().cpu().numpy())

        # Compute EM score.
        exact_match = np.mean((predicted_span_starts == gold_span_starts).all(axis=1))
        self._total_em += exact_match

        # Compute accuracy, F1_m and F_1_a scores.
        predicted_span_starts[predicted_span_starts == -1] = -2  # For comparing predicted and gold arrays
        for predicted_example, gold_example in zip(predicted_span_starts, gold_span_starts):
            n_correct_predicted_per_example = np.intersect1d(predicted_example, gold_example).size
            n_predicted_per_example = predicted_example[predicted_example != -2].size
            n_gold_per_example = gold_example[gold_example != -1].size

            # Update values for F1_m score.
            self._total_precision += float(n_correct_predicted_per_example) / n_gold_per_example
            self._total_recall += float(n_correct_predicted_per_example) / n_predicted_per_example

            # Update values for F1_a score.
            self._total_correct_predicted_span_starts += n_correct_predicted_per_example
            self._total_predicted_span_starts += n_predicted_per_example
            self._total_gold_span_starts += n_gold_per_example

        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float, float]:
        """
        Returns
        -------
        Exact match, accuracy, F1_m and F1_a scores (in that order).
        """
        if self._count == 0:
            exact_match, accuracy, f1_m_score, f1_a_score = 0, 0, 0, 0
        else:
            exact_match = self._total_em / self._count
            accuracy = self._total_precision / self._count
            f1_m_score = self._harmonic_mean(self._total_precision / self._count,
                                             self._total_recall / self._count)
            f1_a_score = self._harmonic_mean(
                self._total_correct_predicted_span_starts / self._total_gold_span_starts,
                self._total_correct_predicted_span_starts / self._total_predicted_span_starts)
        if reset:
            self.reset()
        return exact_match, accuracy, f1_m_score, f1_a_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._total_correct_predicted_span_starts = 0.0
        self._total_predicted_span_starts = 0.0
        self._total_gold_span_starts = 0.0
        self._count = 0

    @staticmethod
    def _harmonic_mean(p, r):
        if p == r == 0:
            return 0
        else:
            return (2 * p * r) / (p + r)
