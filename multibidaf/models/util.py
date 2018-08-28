import torch
from torch.nn.functional import nll_loss

def multi_nll_loss(input, multi_target, weight=None):
    r"""The negative log likelihood loss for multiple targets.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes`.
        multi_target: :math:`(N, M)` where `M = number of targets` and each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in multi_target has to have 0 <= value < C, as before
        >>> # multi_target is of size N x M = 3 x 2
        >>> multi_target = torch.tensor([[1., 3.], [0., 2.], [3., 4.]])
        >>> output = multi_nll_loss(F.log_softmax(input), multi_target)
        >>> output.backward()
    """
    loss_sum = 0
    for target in multi_target.split(split_size=1, dim=1):
        loss_sum += nll_loss(input,
                             target.squeeze(-1),
                             weight,
                             False,
                             -1,
                             True)
    return loss_sum / input.size(0)

