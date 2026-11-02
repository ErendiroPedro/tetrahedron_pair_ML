import torch
import torch.nn.functional as F

EPSILON = 1e-8


def rmsle(predictions, targets):
    log_difference = torch.log(predictions.clamp(min=EPSILON) + 1.0) - torch.log(targets.clamp(min=EPSILON) + 1.0)
    return torch.sqrt((log_difference ** 2).mean() + EPSILON)


def status_and_volume_loss(predictions, targets):
    """Columns are [status logit, volume]; status uses BCE, volume uses RMSLE, weighted equally."""
    status_loss = F.binary_cross_entropy_with_logits(predictions[:, 0], targets[:, 0])
    return 0.5 * rmsle(predictions[:, 1], targets[:, 1]) + 0.5 * status_loss


LOSS_BY_TASK = {
    "IntersectionStatus": F.binary_cross_entropy_with_logits,
    "IntersectionVolume": rmsle,
    "IntersectionStatus_IntersectionVolume": status_and_volume_loss,
}
