import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Acknowledgment: The TransformerEncoder/Decoder and PositionalEncoding concepts
# are heavily based on the original PyTorch tutorials and the "Attention is All You Need" paper.

class PatchEmbed(nn.Module):
    """
    Splits an image into patches and embeds them.
    This is the "tokenizer" for images.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # A convolution layer is a clean way to perform both patching and embedding.
        # It slides a kernel of `patch_size` with a stride of `patch_size`.
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # Input x: (B, C, H, W) -> (B, 1, 128, 128)
        x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w) -> (B, 768, 8, 8)
        x = x.flatten(2)  # (B, embed_dim, n_patches) -> (B, 768, 64)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim) -> (B, 64, 768)
        return x

class TransformerAutoencoder(nn.Module):
    """
    The full Transformer-based Autoencoder.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, latent_dim=2048,
                 embed_dim=768, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

        # --- ENCODER ---
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        # Special learnable token that will be used to summarize the entire sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for each patch + the CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Projection head from Transformer output to the 1D latent space
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        # --- DECODER ---
        # Projection from 1D latent space back to the sequence dimension
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # The decoder needs its own positional embeddings
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        # Standard Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # Final projection layer to get back to the flattened patch dimension
        self.decoder_head = nn.Linear(embed_dim, patch_size**2 * in_chans)

    def encode(self, x):
        # 1. Patchify and embed
        x = self.patch_embed(x) # (B, 64, 768)

        # 2. Prepend CLS token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (B, 1, 768)
        x = torch.cat((cls_token, x), dim=1) # (B, 65, 768)

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Pass through Transformer Encoder
        x = self.transformer_encoder(x) # (B, 65, 768)

        # 5. Extract the CLS token and project to latent space
        cls_output = x[:, 0] # (B, 768)
        latent_vec = self.to_latent(cls_output) # (B, 2048)
        return latent_vec

    def decode(self, z):
        # 1. Project from latent space back to sequence dimension for the decoder
        # This becomes the 'memory' for the decoder's cross-attention
        memory = self.from_latent(z).unsqueeze(1) # (B, 1, 768)
        # We need to attend to this single vector 64 times (once for each output patch)
        memory = memory.repeat(1, self.n_patches, 1) # (B, 64, 768)

        # 2. Create a "dummy" target sequence for the decoder input
        # This will be refined by the decoder. It needs positional info.
        decoder_input = torch.zeros(z.shape[0], self.n_patches, self.embed_dim, device=z.device)
        decoder_input = decoder_input + self.decoder_pos_embed

        # 3. Pass through Transformer Decoder
        x = self.transformer_decoder(tgt=decoder_input, memory=memory) # (B, 64, 768)

        # 4. Project back to patch dimension
        x = self.decoder_head(x) # (B, 64, patch_size*patch_size*C) -> (B, 64, 256)

        # 5. "Un-patchify": Reshape sequence of patches back into an image
        # (B, n_patches, patch_height * patch_width * C) -> (B, C, H, W)
        p = self.patch_size
        h = w = int(math.sqrt(self.n_patches))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        recons_image = x.reshape(shape=(x.shape[0], -1, h * p, w * p))

        return recons_image

    def forward(self, x):
        latent = self.encode(x)
        recons = self.decode(latent)
        return recons, latent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For a quick test, use smaller parameters. For real training, use the defaults.
model = TransformerAutoencoder(
).to(device)

# --- 2. Create a Single Batch of Dummy Data ---
# A single image (batch size = 1) with 1 channel, 128x128
single_image = torch.randn(1, 1, 128, 128).to(device)
out, z = model(single_image)
print("latent shape:", z.shape)  # Should be (1, 2048)
print("Output shape:", out.shape)  # Should be (1, 1,
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_in_million = params / 1e6
print(f"Total trainable parameters: {params_in_million:.2f} million")