"""Stage-1 인코더 = 공식 SpiralNet++ AE의 인코더 절반 + 회귀 헤드.

`SpiralConv` / `Pool` / `SpiralEnblock` 은 벤더링한 공식 코드를 그대로 쓴다
(models/stage1/spiralnet/official/, 출처는 그 폴더의 VENDORED.md).
여기에는 공식에 없는 것만 둔다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage1.spiralnet.official import (  # noqa: F401
    Pool,
    SpiralConv,
    SpiralDeblock,
    SpiralEnblock,
)

__all__ = [
    "Pool",
    "SpiralConv",
    "SpiralDeblock",
    "SpiralEnblock",
    "SpiralEncoder",
    "SpiralDecoder",
    "ClassificationHead",
]


class SpiralEncoder(nn.Module):
    """공식 `AE` 에서 인코더만 떼어낸 것.

    `en_layers` 구성·`forward`·`reset_parameters` 는 공식 AE와 같다. 공식 AE는 디코더에
    업샘플링 행렬(U)이 필요한데, 우리는 mesh → 근육 활성 회귀라 인코더만 쓴다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        latent_channels: int,
        spiral_indices: list[torch.Tensor],
        down_transform: list[torch.Tensor],
    ):
        super().__init__()
        if len(out_channels) != len(spiral_indices) or len(out_channels) != len(down_transform):
            raise ValueError("out_channels, spiral_indices, down_transform length mismatch")

        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.num_vert = self.down_transform[-1].size(0)

        # 공식 AE.__init__ 의 encoder 부분 그대로
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx], self.spiral_indices[idx])
                )
            else:
                self.en_layers.append(
                    SpiralEnblock(
                        out_channels[idx - 1], out_channels[idx], self.spiral_indices[idx]
                    )
                )
        self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # 공식 AE.reset_parameters 그대로: bias 0, 나머지 xavier_uniform
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def _apply(self, fn, *args, **kwargs):
        """`.to(device)` 가 나선 인덱스·다운샘플 행렬까지 옮기게 한다.

        공식 코드는 이 텐서들을 plain attribute 로 들고 있고(`SpiralConv.indices`,
        `AE.down_transform`), main.py 가 모델을 만들기 전에 미리 `.to(device)` 해둔다.
        우리 트레이너는 모델을 만든 뒤에 `model.to(device)` 를 부르므로, 공식 파일을
        건드리지 않고 여기서 같이 옮겨준다.
        """
        super()._apply(fn, *args, **kwargs)
        self.down_transform = [fn(t) for t in self.down_transform]
        self.spiral_indices = [fn(t) for t in self.spiral_indices]
        for m in self.modules():
            if isinstance(m, SpiralConv):
                m.indices = fn(m.indices)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, V, C) → (B, latent_channels). 공식 AE.encoder 그대로."""
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x


class SpiralDecoder(nn.Module):
    """공식 `AE` 에서 디코더만 떼어낸 것.

    `de_layers` 구성과 `forward` 는 공식 `AE.__init__` / `AE.decoder` 를 그대로 옮겼다
    (models/stage1/spiralnet/official/network.py). 인코더가 `SpiralEncoder` 로 분리돼
    있으니 짝을 맞춰 디코더도 분리해 둔다 — mesh→mesh AE (`ModelAE`) 가 쓴다.

    공식과 달라지는 점은 없다. 단, `up_transform` 은 인코더 캐시에 없으므로
    `preprocess.add_up_transforms()` 로 만든 캐시를 써야 한다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        latent_channels: int,
        spiral_indices: list[torch.Tensor],
        up_transform: list[torch.Tensor],
    ):
        super().__init__()
        if len(out_channels) != len(spiral_indices) or len(out_channels) != len(up_transform):
            raise ValueError("out_channels, spiral_indices, up_transform length mismatch")

        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = self.up_transform[-1].size(-1)

        # 공식 AE.__init__ 의 decoder 부분 그대로
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(
                        out_channels[-idx - 1], out_channels[-idx - 1], self.spiral_indices[-idx - 1]
                    )
                )
            else:
                self.de_layers.append(
                    SpiralDeblock(
                        out_channels[-idx], out_channels[-idx - 1], self.spiral_indices[-idx - 1]
                    )
                )
        self.de_layers.append(SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

    def reset_parameters(self):
        # 공식 AE.reset_parameters 그대로
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def _apply(self, fn, *args, **kwargs):
        """`.to(device)` 가 업샘플 행렬·나선 인덱스까지 옮기게 한다 (SpiralEncoder 와 같은 이유)."""
        super()._apply(fn, *args, **kwargs)
        self.up_transform = [fn(t) for t in self.up_transform]
        self.spiral_indices = [fn(t) for t in self.spiral_indices]
        for m in self.modules():
            if isinstance(m, SpiralConv):
                m.indices = fn(m.indices)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, latent_channels) → (B, V, in_channels). 공식 AE.decoder 그대로."""
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x


class ClassificationHead(nn.Module):
    """공식엔 없는 회귀 헤드: latent → 근육 활성 11개.

    활성값이 [0, 1] 이라 Sigmoid 를 둔다. raw 출력이 필요하면 `activation=None`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_outputs: int,
        dropout: float = 0.5,
        activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_outputs)
        self.dropout = float(dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        # 공식 AE.reset_parameters 와 같은 규칙
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(z))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "softmax":
            x = F.log_softmax(x, dim=-1)
        return x
