from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric, CategoricalAccuracy


@Metric.register("multi_categorical_accuracy")
class MultiCategoricalAccuracy(CategoricalAccuracy):
    """
    Multiple Categorical Top-4 accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self, ignore_label: int = -1) -> None:
        super(MultiCategoricalAccuracy, self).__init__(top_k=4)
        self._ignore_label = ignore_label

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, 4). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)

        # Some sanity checks.
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        if gold_labels.dim() != predictions.dim():
            raise ConfigurationError("gold_labels must have dimension == predictions.dimension() but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if gold_labels.size(0) != batch_size:
            raise ConfigurationError('Expected gold_labels batch_size ({}) to match predictions batch_size ({}).'
                             .format(batch_size, gold_labels.size(0)))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Compute the number of correct predictions for each possible number of gold labels.
        correct_per_n_gold_labels = torch.zeros(batch_size, 4)
        for k in range(1, 5):
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(k, predictions.shape[-1]), -1)[1]

            correct_by_k_gold_label = torch.zeros(batch_size)
            for gold_label in gold_labels.split(split_size=1, dim=1):
                correct_by_k_gold_label += top_k.eq(gold_label).sum(dim=1).float()

            correct_by_k_gold_label *= (12 / k)
            correct_per_n_gold_labels[:, k-1] = correct_by_k_gold_label

        # Compute the number of correct predictions.
        max_gold_labels = gold_labels.max(-1)[0].unsqueeze(-1)
        n_gold_labels_mask = gold_labels.eq(max_gold_labels)
        correct = torch.bmm(correct_per_n_gold_labels.view(batch_size, 1, 4).long(),
                            n_gold_labels_mask.view(batch_size, 4, 1).long())

        self.correct_count += correct.sum()
        self.total_count += 12 * batch_size
