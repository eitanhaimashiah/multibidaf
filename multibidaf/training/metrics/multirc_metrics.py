import torch
from typing import List, Tuple, Optional

import numpy as np
from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("multirc")
class MultiRCMetrics(Metric):
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

        # TODO: Update to meaningful names
        # Used for computing F1 score per dataset (F1_a).
        self._total_correct_predicted_span_starts = 0.0
        self._total_predicted_span_starts = 0.0
        self._total_gold_span_starts = 0.0

        self._count = 0

    @overrides
    def __call__(self, scores: List[int],
                 answer_labels: List[int],
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        scores : ``List[int]``
            An array of predicted scores of shape d.
        answer_labels : ``List[int]``
            An array of gold answer labels of shape d.
        """
        scores = np.array(scores)
        answer_labels = np.array(answer_labels)

        # Sanity check
        if scores.shape != answer_labels.shape:
            raise ConfigurationError("scores must have the same shape as answer_labels but "
                                     "found scores of shape {} and answer_labels of shape: {}"
                                     .format(scores.shape, answer_labels.shape))

        # Compute EM score.
        exact_match = (scores == answer_labels).all()
        self._total_em += exact_match

        # Compute accuracy, F1_m and F_1_a scores.
        n_correct_predicted_per_example = (scores * answer_labels).sum()
        n_predicted_per_example = scores.sum()
        n_gold_per_example = answer_labels.sum()

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
            exact_match, accuracy, f1_m_score, f1_a_score = 0.0, 0.0, 0.0, 0.0
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
