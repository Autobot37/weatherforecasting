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
        latent_channels: int = 16,
        groups: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 64,
        nlayers: int = 4
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder down to 16×16 feature map
        self.enc = nn.Sequential(
            EncBlock(in_channels, 256, num_blocks=4, groups=groups),
            EncBlock(256, 512, num_blocks=4, groups=groups),
            EncBlock(512, 1024, num_blocks=4, groups=groups),
            nn.Conv2d(1024, latent_channels, 1)
        )
        # Learned pos‑emb for 16×16
        self.pos_emb = nn.Parameter(torch.randn(1, latent_channels, 16, 16))

        # Single TF layer to mix spatial tokens in decoder
        # We treat the 16×16 map as a sequence of 256 tokens
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=latent_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.tf = nn.TransformerEncoder(encoder_layers, nlayers)

        # Conv‑decoder from 16×16 map → 128×128 output
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, 1),
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
        return z.flatten(1)  # (B, 16*16*latent_channels)

    def decode(self, z_flat):
        B = z_flat.size(0)
        # 1) reshape to (B, C, H, W)
        z = z_flat.view(B, self.latent_channels, 16, 16)

        # 2) flatten spatial → (B, 256, C)
        seq = z.flatten(2).transpose(1,2)

        # 3) one TF block
        seq = self.tf(seq)  # (B, 256, C)

        # 4) reshape back → (B, C, 16,16)
        z2 = seq.transpose(1,2).view(B, self.latent_channels, 16, 16)

        # 5) conv decode → (B,1,128,128)
        return self.act(self.dec(z2))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# Quick sanity check
if __name__ == "__main__":
    net = PosAwareAE_TF().cuda()
    for name, p in net.named_modules():
        params = sum(p.numel() for p in p.parameters() if p.requires_grad)
        print(f"{name}: {params:.2f} params")
    x = torch.randn(2,1,128,128).cuda()
    y, z = net(x)
    print("output:", y.shape)        # (2,1,128,128)
    print("latent:", z.shape)        # (2,4096)
    print("params: %.1fM" % (sum(p.numel() for p in net.parameters())/1e6))
