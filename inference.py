"""CPU inference for the 1-bit ResNet18."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load as _jit_load

_REPO_ROOT = Path(__file__).resolve().parent
_KERNEL_SRC = _REPO_ROOT / "bnn_kernel.cpp"


def _build_extension():
    return _jit_load(
        name="bnn_kernel",
        sources=[str(_KERNEL_SRC)],
        extra_cflags=["-O3", "-ffast-math", "-std=c++17"],
        verbose=False,
    )


bnn = _build_extension()


_MSB_SHIFTS = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32)


def pack_bits_msb(bits_u8: torch.Tensor) -> torch.Tensor:
    """Pack a [..., 8*K_bytes] uint8 into [..., K_bytes] uint8."""
    if bits_u8.dtype != torch.uint8:
        bits_u8 = bits_u8.to(torch.uint8)
    last = bits_u8.shape[-1]
    assert last % 8 == 0, f"last dim must be a multiple of 8, got {last}"
    grouped = bits_u8.view(*bits_u8.shape[:-1], last // 8, 8).to(torch.int32)
    packed = (grouped << _MSB_SHIFTS).sum(dim=-1).to(torch.uint8)
    return packed



def load_packed_checkpoint(path: str | os.PathLike) -> Dict[str, torch.Tensor]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def unpack_binary_weights(packed_dict: Dict[str, torch.Tensor]):
    """Split a packed checkpoint into (binary_layer_info, passthrough_state_dict)."""
    binary_info: Dict[str, dict] = {}
    passthrough: Dict[str, torch.Tensor] = {}

    module_paths = set()
    for k in packed_dict:
        if k.endswith(".weight_packed"):
            module_paths.add(k[: -len(".weight_packed")])

    for mod_path in module_paths:
        base = mod_path + ".weight"
        packed = packed_dict[base + "_packed"]
        scale = packed_dict[base + "_scale"]
        shape = tuple(packed_dict[base + "_shape"])

        if isinstance(packed, np.ndarray):
            packed_t = torch.from_numpy(packed.astype(np.uint8)).contiguous()
        else:
            packed_t = packed.to(torch.uint8).contiguous()

        M = shape[0]
        numel = int(np.prod(shape))
        K = numel // M  # logical K = C * kH * kW
        assert numel % M == 0, "packed weight numel must be divisible by M"
        assert K % 8 == 0, f"K={K} for {base} is not a multiple of 8"
        K_bytes = K // 8
        assert packed_t.numel() == M * K_bytes, (
            f"packed bytes mismatch for {base}: "
            f"{packed_t.numel()} vs {M}*{K_bytes}"
        )

        packed_mk = packed_t.view(M, K_bytes).contiguous()

        bits = np.unpackbits(packed_t.numpy())[:numel].reshape(shape)
        weight_pm1 = torch.from_numpy(
            (bits.astype(np.float32) * 2.0 - 1.0)
        ).contiguous()

        if isinstance(scale, np.ndarray):
            scale_t = torch.from_numpy(scale).to(torch.float32)
        else:
            scale_t = scale.to(torch.float32)

        binary_info[mod_path] = {
            "packed": packed_mk,
            "weight_pm1": weight_pm1,
            "scale": scale_t,
            "shape": shape,
        }

    for k, v in packed_dict.items():
        if k.endswith("_packed") or k.endswith("_scale") or k.endswith("_shape"):
            continue
        passthrough[k] = v

    return binary_info, passthrough



class FloatSignConv2d(nn.Module):
    """Convolution with +/-1 float weights. Used when inputs are NOT binary."""

    def __init__(self, weight_pm1: torch.Tensor, stride=1, padding=0):
        super().__init__()
        self.register_buffer("weight", weight_pm1)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, None, self.stride, self.padding)


class PackedBinaryConv2d(nn.Module):
    """Binary convolution on +/-1 inputs using the NEON XNOR-popcount kernel."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        shape: Tuple[int, int, int, int],
        stride=1,
        padding=0,
    ):
        super().__init__()
        M, C, kH, kW = shape
        self.out_channels = M
        self.in_channels = C
        self.kernel_size = (kH, kW)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.K = C * kH * kW
        assert self.K % 8 == 0, f"K={self.K} must be multiple of 8"
        assert C % 8 == 0, f"C={C} must be multiple of 8 for channel-inner layout"
        self.K_bytes = self.K // 8
        assert packed_weight.shape == (M, self.K_bytes), (
            f"packed_weight shape {packed_weight.shape} != {(M, self.K_bytes)}"
        )
        packed_khwc = bnn.repack_weight_khwc(
            packed_weight.contiguous(), C, kH, kW
        )
        self.register_buffer("packed_weight", packed_khwc.contiguous())
        self._padded = self.padding != (0, 0)
        self._mask_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_mask(self, N: int, H: int, W: int):
        key = (N, H, W)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        mask, k_valid = bnn.build_mask(
            N, self.in_channels, H, W, kH, kW, pH, pW, sH, sW
        )
        self._mask_cache[key] = (mask, k_valid)
        return mask, k_valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        assert C == self.in_channels, f"expected C={self.in_channels}, got {C}"

        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        packed_in = bnn.im2col_sign_pack(x, kH, kW, pH, pW, sH, sW)

        if self._padded:
            packed_mask, k_valid = self._get_mask(N, H, W)
            out_flat = bnn.bgemm_neon_masked(
                packed_in, packed_mask, self.packed_weight, k_valid
            )
        else:
            out_flat = bnn.bgemm_neon(packed_in, self.packed_weight, self.K)

        L = H_out * W_out
        out = out_flat.view(N, L, self.out_channels).transpose(1, 2).contiguous()
        return out.view(N, self.out_channels, H_out, W_out)


class InferenceBasicBlock(nn.Module):
    """Topology-identical to model.BinaryBasicBlock at inference time."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        binary_info: Dict[str, dict],
        prefix: str,
        first_block_of_layer1: bool = False,
    ):
        super().__init__()
        conv1_info = binary_info[f"{prefix}.conv1"]
        conv2_info = binary_info[f"{prefix}.conv2"]

        if first_block_of_layer1:
            self.conv1 = FloatSignConv2d(
                conv1_info["weight_pm1"], stride=stride, padding=1
            )
        else:
            self.conv1 = PackedBinaryConv2d(
                conv1_info["packed"], conv1_info["shape"], stride=stride, padding=1
            )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = PackedBinaryConv2d(
            conv2_info["packed"], conv2_info["shape"], stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(planes)

        needs_shortcut = stride != 1 or in_planes != self.expansion * planes
        if needs_shortcut:
            sc_info = binary_info[f"{prefix}.shortcut.0"]
            self.shortcut = nn.Sequential(
                PackedBinaryConv2d(
                    sc_info["packed"], sc_info["shape"], stride=stride, padding=0
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Sequential()

    @staticmethod
    def _sign_pm1(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self._sign_pm1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self._sign_pm1(out)
        return out


class BinaryResNet18Inference(nn.Module):
    """CPU inference version of model.BinaryResNet18 using packed 1-bit weights."""

    def __init__(self, num_classes: int = 35):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1: nn.Sequential
        self.layer2: nn.Sequential
        self.layer3: nn.Sequential
        self.layer4: nn.Sequential

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        binary_info: Dict[str, dict],
        layer_prefix: str,
        is_layer1: bool,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        cur_in = in_planes
        for idx, s in enumerate(strides):
            block = InferenceBasicBlock(
                cur_in,
                planes,
                s,
                binary_info,
                prefix=f"{layer_prefix}.{idx}",
                first_block_of_layer1=(is_layer1 and idx == 0),
            )
            layers.append(block)
            cur_in = planes * InferenceBasicBlock.expansion
        return nn.Sequential(*layers), cur_in

    @classmethod
    def from_packed_checkpoint(
        cls, path: str | os.PathLike, num_classes: int = 35
    ) -> "BinaryResNet18Inference":
        packed = load_packed_checkpoint(path)
        binary_info, passthrough = unpack_binary_weights(packed)

        model = cls(num_classes=num_classes)

        in_planes = 64
        model.layer1, in_planes = cls._make_layer(
            in_planes, 64, 2, 1, binary_info, "layer1", is_layer1=True
        )
        model.layer2, in_planes = cls._make_layer(
            in_planes, 128, 2, 2, binary_info, "layer2", is_layer1=False
        )
        model.layer3, in_planes = cls._make_layer(
            in_planes, 256, 2, 2, binary_info, "layer3", is_layer1=False
        )
        model.layer4, in_planes = cls._make_layer(
            in_planes, 512, 2, 2, binary_info, "layer4", is_layer1=False
        )

        missing, unexpected = model.load_state_dict(passthrough, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

        expected_binary_keys = {
            f"{b}.packed_weight" for b in binary_info.keys()
        } | {"layer1.0.conv1.weight"}
        real_missing = [k for k in missing if k not in expected_binary_keys]
        if real_missing:
            raise RuntimeError(f"Missing keys not supplied by checkpoint: {real_missing}")

        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def build_float_reference(path: str | os.PathLike, num_classes: int = 35) -> nn.Module:
    """Return a model.BinaryResNet18 with weights reconstructed from packed bits."""
    from model import BinaryResNet18

    packed = load_packed_checkpoint(path)
    binary_info, passthrough = unpack_binary_weights(packed)

    model = BinaryResNet18(num_classes=num_classes)
    sd = dict(passthrough)
    for mod_path, info in binary_info.items():
        sd[f"{mod_path}.weight"] = info["weight_pm1"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    return model
