import torch
import torch.nn as nn

class GlobalCrossEncode(nn.Module):
    """
    Cross-attention that collapses a sequence of d_token tokens → one d_latent vector.
    q: (B, 1, d_latent), kv: (B, L, d_token)
    returns: (B, d_latent)
    """
    def __init__(self, d_token, d_latent, n_heads=8):
        super().__init__()
        assert d_latent % n_heads == 0 and d_token % n_heads == 0
        self.nh     = n_heads
        self.dh_q   = d_latent  // n_heads
        self.dh_kv  = d_token   // n_heads
        self.scale  = self.dh_q ** -0.5

        self.q_proj  = nn.Linear(d_latent, d_latent)
        self.kv_proj = nn.Linear(d_token, 2 * d_latent)
        self.out     = nn.Linear(d_latent, d_latent)

    def forward(self, q, kv):
        B, L, _ = kv.shape
        # 1) project q→(B,1,nh,dh_q), kv→(B,L,2,nh,dh_q)
        q  = self.q_proj(q) \
                .view(B, 1, self.nh, self.dh_q) \
                .transpose(1, 2)            # → (B,nh,1,dh_q)
        kv = self.kv_proj(kv) \
                .view(B, L, 2, self.nh, self.dh_q)
        k  = kv[...,0,:,:].transpose(1,2)   # → (B,nh,L,dh_q)
        v  = kv[...,1,:,:].transpose(1,2)   # → (B,nh,L,dh_q)

        # 2) attention weights (B,nh,1,L)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)

        # 3) attend v → (B,nh,1,dh_q) → merge heads → (B,1,d_latent)
        out = (attn @ v) \
                .transpose(1,2) \
                .contiguous() \
                .view(B, 1, -1)
        return self.out(out).squeeze(1)      # → (B, d_latent)

class GlobalCrossDecode(nn.Module):
    """
    Cross-attention that expands one d_latent vector → sequence of d_token tokens.
    q: (B, L, d_token), kv: (B, 1, d_latent)
    returns: (B, L, d_token)
    """
    def __init__(self, d_token, d_latent, n_heads=8):
        super().__init__()
        assert d_latent % n_heads == 0 and d_token % n_heads == 0
        self.nh     = n_heads
        self.dh_q   = d_token   // n_heads
        self.dh_kv  = d_latent  // n_heads
        self.scale  = self.dh_kv ** -0.5

        self.q_proj  = nn.Linear(d_token, d_token)
        self.kv_proj = nn.Linear(d_latent, 2 * d_token)
        self.out     = nn.Linear(d_token, d_token)

    def forward(self, q, kv):
        B, L, _ = q.shape
        # 1) project q→(B,L,nh,dh_q), kv→(B,1,2,nh,dh_q)
        q  = self.q_proj(q) \
                .view(B, L, self.nh, self.dh_q) \
                .transpose(1, 2)            # → (B,nh,L,dh_q)
        kv = self.kv_proj(kv) \
                .view(B, 1, 2, self.nh, self.dh_q)
        k  = kv[...,0,:,:].transpose(1,2)   # → (B,nh,1,dh_q)
        v  = kv[...,1,:,:].transpose(1,2)   # → (B,nh,1,dh_q)

        # 2) attention weights (B,nh,L,1)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)

        # 3) attend v → (B,nh,L,dh_q) → merge heads → (B,L,d_token)
        out = (attn @ v) \
                .transpose(1,2) \
                .contiguous() \
                .view(B, L, -1)
        return self.out(out)                # → (B, L, d_token)

class AE_ViT_2048(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) constants
        img, patch, ch = 128, 16, 1
        seq            = img // patch      # 8
        n_patches      = seq * seq        # 64
        d_token, d_latent = 512, 2048
        depth_enc, depth_dec, heads = 6, 6, 8

        # 2) store dims
        self.seq, self.d_token = seq, d_token
        self.d_latent          = d_latent

        # 3) patch embedding
        self.patch_embed = nn.Conv2d(ch, d_token, patch, patch)

        # 4) positional tokens (64 of them, dim=32)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_token))

        # 5) encoder: 6× Transformer blocks
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=heads,
            dim_feedforward=4*d_token,
            dropout=0.1, activation='gelu',
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth_enc)

        # 6) cross-attention → latent
        self.query_vec = nn.Parameter(torch.randn(1, 1, d_latent))
        self.to_latent = GlobalCrossEncode(d_token, d_latent, n_heads=heads)

        # 7) cross-attention ← latent
        self.dec_queries = nn.Parameter(torch.randn(1, n_patches, d_token))
        self.from_latent = GlobalCrossDecode(d_token, d_latent, n_heads=heads)

        # 8) decoder: 6× Transformer blocks
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=heads,
            dim_feedforward=4*d_token,
            dropout=0.1, activation='gelu',
            batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, depth_dec)

        # 9) unpatchify back to image
        self.unpatch = nn.ConvTranspose2d(
            d_token, ch, patch, patch)

    def forward(self, x):
        B = x.size(0)

        # — Encoder path —
        # a) patches → (B,64,32)
        z = self.patch_embed(x)                  # (B,32,8,8)
        z = z.flatten(2).transpose(1,2)          # → (B,64,32)
        z = z + self.pos_embed                   # add position
        z = self.encoder(z)                      # → (B,64,32)

        # b) collapse to latent: (B,1,2048) → (B,2048)
        q = self.query_vec.expand(B, -1, -1)     # → (B,1,2048)
        latent = self.to_latent(q, z)            # → (B,2048)

        # — Decoder path —
        # c) prepare token queries (64,32) and expand latent to (1,2048)
        dec_q = self.dec_queries.expand(B, -1, -1)     # → (B,64,32)
        kv    = latent.unsqueeze(1)                   # → (B,1,2048)

        # d) cross-attention to get (B,64,32)
        z_dec = self.from_latent(dec_q, kv)            # → (B,64,32)
        z_dec = z_dec + self.pos_embed                 # re-add positional

        # e) transformer decode
        z_dec = self.decoder(z_dec)                    # → (B,64,32)

        # f) unpatchify back to image
        z_dec = z_dec.transpose(1,2).view(B, -1, self.seq, self.seq)
        out   = self.unpatch(z_dec)                    # → (B,1,128,128)

        return out, latent

model = AE_ViT_2048()
params_million_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model AE_ViT_2048 has {params_million_count:.2f} million parameters.")
x = torch.randn(2, 1, 128, 128)  # Example input
out, latent = model(x)
print(f"Output shape: {out.shape}, Latent shape: {latent.shape}")
print(f"Latent vector size: {model.d_latent}, Token size: {model.d_token}")