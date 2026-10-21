import torch
from torch import nn


class CTCLossWithSmoothing(nn.Module):

    def __init__(self, blank_id: int, smoothing_weight: float = 0.1, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=zero_infinity)
        self.smoothing_weight = smoothing_weight

    def forward(
        self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:

        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=2)

        max_len, batch_size = log_probs.shape[:2]
        time_steps = torch.arange(max_len, device=log_probs.device).unsqueeze(1).expand(-1, batch_size)
        mask = time_steps < input_lengths.unsqueeze(0)

        valid_entropy = entropy * mask
        avg_entropy = valid_entropy.sum() / mask.sum()

        return ctc_loss - (self.smoothing_weight * avg_entropy)
