import torch
import torch.nn as nn

# ------------------------------------------------------------
# Pre-Act Bottleneck (no SE, simple, stable)
# channels -> channels, 3×3 grouped conv
# ------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        mid = channels // 4
        # ensure groups divides mid
        g = min(groups, mid)
        assert mid % g == 0, f"groups ({g}) must divide mid channels ({mid})"
        self.f = nn.Sequential(
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False)
        )

    def forward(self, x):
        return x + self.f(x)

# ------------------------------------------------------------
# Encoder block: stride-2 conv + N Bottlenecks
# ------------------------------------------------------------
class EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, groups: int = 8):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.res = nn.Sequential(*[Bottleneck(out_ch, groups=groups) for _ in range(num_blocks)])

    def forward(self, x):
        return self.res(self.down(x))

# ------------------------------------------------------------
# Decoder block: stride-2 transposed conv + N Bottlenecks
# ------------------------------------------------------------
class DecBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, groups: int = 8):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.res = nn.Sequential(*[Bottleneck(out_ch, groups=groups) for _ in range(num_blocks)])

    def forward(self, x):
        return self.res(self.up(x))

# ------------------------------------------------------------
# Full AE with optional activation
# ------------------------------------------------------------
# class PosAwareAutoEncoder(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 1,
#         latent_channels: int = 64,
#         groups: int = 8,
#     ):
#         super().__init__()
#         self.latent_channels = latent_channels

#         # ------------- Encoder -------------
#         self.enc = nn.Sequential(
#             EncBlock(in_channels, 128, num_blocks=2, groups=groups),
#             EncBlock(128, 256, num_blocks=3, groups=groups),
#             EncBlock(256, 512, num_blocks=4, groups=groups),
#             EncBlock(512, 1024, num_blocks=4, groups=groups),
#             nn.Conv2d(1024, latent_channels, 1)
#         )

#         # ------------- Positional embedding -------------
#         self.pos_emb = nn.Parameter(torch.randn(latent_channels, 8, 8))

#         self.to_timeseries = nn.Linear(4096, 4096)
#         self.from_timeseries = nn.Linear(4096, 4096)

#         # ------------- Decoder -------------
#         self.dec = nn.Sequential(
#             nn.Conv2d(latent_channels, 1024, 1),
#             DecBlock(1024, 512, num_blocks=4, groups=groups),
#             DecBlock(512, 256, num_blocks=4, groups=groups),
#             DecBlock(256, 128, num_blocks=3, groups=groups),
#             DecBlock(128,  64, num_blocks=2, groups=groups),
#             nn.Conv2d(64, in_channels, 3, padding=1)
#         )

#         self.act = nn.Sigmoid()

#     def encode(self, x):
#         z = self.enc(x)
#         z = z + self.pos_emb
#         z = z.flatten(1)
#         z = self.to_timeseries(z)
#         return z

#     def decode(self, z_flat):
#         z_flat = self.from_timeseries(z_flat)
#         z = z_flat.view(-1, self.latent_channels, 8, 8)
#         return self.act(self.dec(z))

#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z), z

import torch
import torch.nn as nn
import math

# ---------------------------------------------------------
# Learnable 2-D positional code that will be concatenated
# ---------------------------------------------------------
class CoordEmbedding(nn.Module):
    """
    Returns (B, 2, H, W) tensor with normalized (x, y) coords.
    """
    def __init__(self, h=8, w=8):
        super().__init__()
        y, x = torch.meshgrid(torch.linspace(-1, 1, h),
                              torch.linspace(-1, 1, w),
                              indexing='ij')
        self.register_buffer('coords', torch.stack([x, y], dim=0)[None])  # (1,2,H,W)

    def forward(self, b):
        return self.coords.expand(b, -1, -1, -1)  # (B,2,H,W)

# ---------------------------------------------------------
# PosAwareAutoEncoder with 1-D latent and spatial indices
# ---------------------------------------------------------
class PosAwareAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        groups: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.h = self.w = 8          # fixed spatial size in latent
        flat_len = latent_channels * self.h * self.w

        # ------------- Encoder -------------
        self.enc = nn.Sequential(
            EncBlock(in_channels, 128, num_blocks=2, groups=groups),
            EncBlock(128, 256, num_blocks=3, groups=groups),
            EncBlock(256, 512, num_blocks=4, groups=groups),
            EncBlock(512, 1024, num_blocks=4, groups=groups),
            nn.Conv2d(1024, latent_channels, 1)
        )

        # ------------- Spatial coordinate code -------------
        self.coord_emb = CoordEmbedding(self.h, self.w)  # (B,2,8,8)

        # ------------- 1-D sequence model (lightweight) -----
        #   input:  (B, 8*8, latent_channels + 2)
        #   output: (B, 8*8, latent_channels)
        d_model = latent_channels + 2
        self.seq_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=11,
                dim_feedforward=4*d_model,
                dropout=0.0,
                batch_first=True
            ),
            num_layers=2
        )
        self.to_latent = nn.Linear(d_model, latent_channels)   # project back
        self.from_latent = nn.Linear(latent_channels, d_model) # project forward

        self.to_ = nn.Linear(4096, 4096)
        self.from_ = nn.Linear(4096, 4096)
        # ------------- Decoder -------------
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, 1),
            DecBlock(1024, 512, num_blocks=4, groups=groups),
            DecBlock(512, 256, num_blocks=4, groups=groups),
            DecBlock(256, 128, num_blocks=3, groups=groups),
            DecBlock(128,  64, num_blocks=2, groups=groups),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
        self.act = nn.Sigmoid()

    # ------------------------------------------------------------------
    def encode(self, x):
        # 1) 2-D feature map
        z = self.enc(x)                                   # (B,C,8,8)
        coords = self.coord_emb(z.size(0))                # (B,2,8,8)

        # 2) Concatenate channel + coords
        z = torch.cat([z, coords], dim=1)                 # (B,C+2,8,8)

        # 3) Row-major flatten → 1-D sequence
        z = z.flatten(2).transpose(1, 2)                  # (B, 64, C+2)

        # 4) 1-D sequence processing
        z = self.seq_model(z)                             # (B,64,C+2)
        z = self.to_latent(z)                             # (B,64,C)

        # 5) Flatten to 1-D latent you want
        z = z.flatten(1)                                  # (B, 64*C)
        z = self.to_(z)
        return z

    # ------------------------------------------------------------------
    def decode(self, z_flat):
        # 1) Unflatten
        b = z_flat.size(0)
        z = self.from_(z_flat)
        z = z.view(b, self.h*self.w, self.latent_channels)

        # 2) 1-D processing again
        z = self.from_latent(z)                           # (B,64,C+2)
        z = self.seq_model(z)                             # (B,64,C+2)
        z = self.to_latent(z)                             # (B,64,C)

        # 3) Undo row-major permutation
        z = z.transpose(1, 2).view(b, self.latent_channels,
                                   self.h, self.w)        # (B,C,8,8)

        # 4) Decode to image
        return self.act(self.dec(z))

    # ------------------------------------------------------------------
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    net = PosAwareAutoEncoder()
    x = torch.randn(2, 1, 128, 128)
    y, z = net(x)
    print("output:", y.shape)
    print("latent", z.shape)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"trainable params: {params:.1f} M")
