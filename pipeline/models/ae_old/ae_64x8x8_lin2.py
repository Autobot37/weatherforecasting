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
class PosAwareAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        groups: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # ------------- Encoder -------------
        self.enc = nn.Sequential(
            EncBlock(in_channels, 128, num_blocks=2, groups=groups),
            EncBlock(128, 256, num_blocks=3, groups=groups),
            EncBlock(256, 512, num_blocks=4, groups=groups),
            EncBlock(512, 1024, num_blocks=4, groups=groups),
            nn.Conv2d(1024, latent_channels, 1)
        )

        # ------------- Positional embedding -------------
        self.pos_emb = nn.Parameter(torch.randn(latent_channels, 8, 8))

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

    def encode(self, x):
        z = self.enc(x)
        z = z + self.pos_emb
        return z.flatten(1)

    def decode(self, z_flat):
        z = z_flat.view(-1, self.latent_channels, 8, 8)
        return self.act(self.dec(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    net = PosAwareAutoEncoder()
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    print("output:", y.shape)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"trainable params: {params:.1f} M")
