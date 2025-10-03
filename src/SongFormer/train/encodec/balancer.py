from collections import defaultdict
import typing as tp

import torch
from torch import autograd

from .distrib import average_metrics


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(
        metrics: tp.Dict[str, tp.Any], weight: float = 1
    ) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}

    return _update


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(
        self,
        weights: tp.Dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
        monitor: bool = False,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def cal_mix_loss(
        self,
        losses: tp.Dict[str, torch.Tensor],
        params: tp.List[torch.Tensor],
        accelerator,
    ):
        assert accelerator is not None, "accelerator is None"
        norms = {}
        grads = {}
        # import pdb

        # pdb.set_trace()
        for name, loss in losses.items():
            if self.per_batch_item and loss.dim() > 0:
                # 假定 loss.shape[0] == batch_size
                norm_list = []
                batch_size = loss.shape[0]
                for i in range(batch_size):
                    loss_item = loss[i]
                    grads_tuple = autograd.grad(
                        loss_item, params, retain_graph=True, allow_unused=True
                    )
                    flat_grad = torch.cat(
                        [g.view(-1) for g in grads_tuple if g is not None]
                    )
                    norm_item = flat_grad.norm()
                    norm_list.append(norm_item)
                norm = torch.stack(norm_list).mean()
            else:
                grads_tuple = autograd.grad(
                    loss, params, retain_graph=True, allow_unused=True
                )
                flat_grad = torch.cat(
                    [g.view(-1) for g in grads_tuple if g is not None]
                )
                norm = flat_grad.norm()
            norms[name] = norm

        count = 1
        if self.per_batch_item and loss.dim() > 0:
            count = len(norm_list)

        avg_norms = average_metrics(
            self.averager(norms), count, accelerator=accelerator
        )
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f"ratio_{k}"] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_loss: tp.Any = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                loss_scaled = losses[name] * scale
            else:
                loss_scaled = self.weights[name] * losses[name]
            out_loss += loss_scaled
        return out_loss


def test():
    from torch.nn import functional as F

    x = torch.zeros(1, requires_grad=True)
    one = torch.ones_like(x)
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {"1": loss_1, "2": loss_2}

    balancer = Balancer(weights={"1": 1, "2": 1}, rescale_grads=False)
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.0)), x.grad

    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {"1": loss_1, "2": loss_2}
    x.grad = None
    balancer = Balancer(weights={"1": 1, "2": 1}, rescale_grads=True)
    balancer.backward({"1": loss_1, "2": loss_2}, x)
    assert torch.allclose(x.grad, torch.tensor(0.0)), x.grad


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    test()
