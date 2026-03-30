import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Connect4


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self, rows: int = 6, cols: int = 7, num_res_blocks: int = 5, num_channels: int = 64):
        super().__init__()
        self.rows = rows
        self.cols = cols

        # Input conv
        self.input_conv = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, cols)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared trunk
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head: (batch, cols) logits
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head: (batch, 1) in [-1, 1]
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    @torch.no_grad()
    def predict(self, state: Connect4, device: str = "cpu") -> tuple[np.ndarray, float]:
        """Run inference on a single game state.

        Returns (policy, value) as numpy arrays.
        Policy is a probability distribution over columns (masked for valid moves).
        Value is a scalar in [-1, 1] from current player's perspective.
        """
        self.eval()
        encoded = torch.tensor(state.encode(), dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = self(encoded)

        # Mask invalid moves and softmax
        valid = state.get_valid_moves()
        logits = logits.squeeze(0).cpu().numpy()
        logits[~valid] = -np.inf
        # Stable softmax
        logits = logits - logits[valid].max()
        exp = np.exp(logits)
        policy = exp / exp.sum()

        return policy, value.item()
