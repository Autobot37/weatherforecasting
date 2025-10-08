import torch
import torch.nn as nn

# ------------------------------------------------------------
# Pre‑Act Bottleneck (unchanged)
# ------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        mid = channels // 4
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
# Encoder / Decoder blocks (unchanged)
# ------------------------------------------------------------
class EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, groups: int = 8):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.res = nn.Sequential(*[Bottleneck(out_ch, groups) for _ in range(num_blocks)])
    def forward(self, x):
        return self.res(self.down(x))

class DecBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, groups: int = 8):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.res = nn.Sequential(*[Bottleneck(out_ch, groups) for _ in range(num_blocks)])
    def forward(self, x):
        return self.res(self.up(x))

# ------------------------------------------------------------
# Pos‑Aware AE + one Transformer layer in decoder
# ------------------------------------------------------------
class PosAwareAE_TF(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        groups: int = 8,
        latent_dim: int = 2048
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder down to 16×16 feature map
        self.enc = nn.Sequential(
            EncBlock(in_channels, 256, num_blocks=4, groups=groups),
            EncBlock(256, 512, num_blocks=4, groups=groups),
            EncBlock(512, 1024, num_blocks=4, groups=groups),
            EncBlock(1024, 1024, num_blocks=4, groups=groups),
            nn.Conv2d(1024, latent_channels, 1)
        )
        # Learned pos‑emb for 16×16
        self.pos_emb = nn.Parameter(torch.randn(1, latent_channels, 8, 8))

        self.to_latent = nn.Linear(8 * 8 * latent_channels, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 8 * 8 * latent_channels)

        self.tf_encoder = nn.TransformerEncoderLayer(
            d_model=latent_channels, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.tf = nn.TransformerEncoder(self.tf_encoder, num_layers=8)

        # Conv‑decoder from 16×16 map → 128×128 output
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, 1),
            DecBlock(1024, 1024, num_blocks=4, groups=groups),
            DecBlock(1024, 512, num_blocks=4, groups=groups),
            DecBlock(512, 256, num_blocks=4, groups=groups),
            DecBlock(256, 128, num_blocks=4, groups=groups),
            nn.Conv2d(128, in_channels, 3, padding=1)
        )
        self.act = nn.Sigmoid()

    def encode(self, x):
        # x -> (B, latent_channels, 16,16)
        z = self.enc(x)
        z = z + self.pos_emb
        z = z.flatten(1)  # (B, 16*16*latent_channels)
        z = self.to_latent(z)  # (B, latent_dim)
        return z

    def decode(self, z_flat):
        B = z_flat.size(0)
        z = self.from_latent(z_flat)  # (B, 16*16*latent_channels)
        # 1) reshape to (B, C, H, W)
        z = z.view(B, self.latent_channels, 8, 8)

        z_tokens = z.flatten(2).transpose(1, 2)  # (B, 64, C)
        z_tokens = self.tf(z_tokens)            # (B, 64, C)
        z = z_tokens.transpose(1, 2).view(B, self.latent_channels, 8, 8)  # back to (B, C, 8, 8)

        # 5) conv decode → (B,1,128,128)
        return self.act(self.dec(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# Quick sanity check
if __name__ == "__main__":
    net = PosAwareAE_TF().cuda()
    x = torch.randn(2,1,128,128).cuda()
    y, z = net(x)
    print("output:", y.shape)        # (2,1,128,128)
    print("latent:", z.shape)        # (2,4096)
    print("params: %.1fM" % (sum(p.numel() for p in net.parameters())/1e6))
