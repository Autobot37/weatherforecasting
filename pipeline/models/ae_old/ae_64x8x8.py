import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Residual Block (Modern)
# ----------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)

        # Match shortcut channels
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


# ----------------------
# Encoder
# ----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=256):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.block1 = ResBlock(base_channels, base_channels * 2, stride=2)   # 128->64
        self.block2 = ResBlock(base_channels * 2, base_channels * 4, stride=2)  # 64->32
        self.block3 = ResBlock(base_channels * 4, base_channels * 8, stride=2)  # 32->16
        self.block4 = ResBlock(base_channels * 8, base_channels * 16, stride=2)  # 16->8
        
        self.final = nn.Conv2d(base_channels * 16, 64, 1)  # Compress to 64 channels

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)


# ----------------------
# Decoder
# ----------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=256):
        super().__init__()
        self.proj = nn.Conv2d(64, base_channels * 16, 1)

        self.block0 = ResBlock(base_channels * 16, base_channels * 8)
        self.up0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)  # 16->32

        self.block1 = ResBlock(base_channels * 8, base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)  # 16->32

        self.block2 = ResBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)  # 32->64

        self.block3 = ResBlock(base_channels * 2, base_channels)
        self.up3 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)  # 64->128

        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.up0(self.block0(x))
        x = self.up1(self.block1(x))
        x = self.up2(self.block2(x))
        x = self.up3(self.block3(x))
        return torch.sigmoid(self.final(x))

# ----------------------
# Full Autoencoder
# ----------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(base_channels=64)
        self.decoder = Decoder(base_channels=64)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# ----------------------
# Test + Param Count
# ----------------------
if __name__ == "__main__":
    model = AutoEncoder()
    x = torch.randn(2, 1, 128, 128)
    recon, latent = model(x)
    print("Input:", x.shape)
    print("Latent:", latent.shape)
    print("Reconstruction:", recon.shape)
    print("Params:", sum(p.numel() for p in model.parameters())/1e6, "M")
