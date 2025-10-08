import torch
import torch.nn as nn

class PositionalAutoencoder(nn.Module):
    """
    An autoencoder that uses positional embeddings and has a flexible channel depth.
    """
    def __init__(self, base_channels=16, spatial_size=8):
        super().__init__()
        
        # Final channel depth is derived from the base_channels
        final_channels = base_channels * 4 # e.g., 16 * 4 = 64
        
        self.channels = final_channels
        self.spatial_size = spatial_size
        self.sequence_length = spatial_size * spatial_size
        self.latent_dim = self.sequence_length * self.channels

        # =================== 1. ENCODER ===================
        # Channel depth now scales with base_channels
        self.encoder = nn.Sequential(
            # Input: (N, 1, 128, 128)
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),             # -> (N, base_ch, 64, 64)
            nn.ReLU(True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # -> (N, base_ch*2, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(base_channels * 2, self.channels, kernel_size=3, stride=2, padding=1), # -> (N, base_ch*4, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1),   # -> (N, base_ch*4, 8, 8)
        )

        # =================== 2. POSITIONAL EMBEDDING ===================
        self.pos_embedding = nn.Parameter(torch.randn(self.sequence_length, self.channels))

        # =================== 3. DECODER ===================
        self.decoder_input = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1),      # -> (N, base_ch*4, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channels, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (N, base_ch*2, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (N, base_ch, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),               # -> (N, 1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- ENCODING ---
        features = self.encoder(x)
        N, C, H, W = features.shape
        
        # Reshape and permute to create a sequence: (N, C, H*W) -> (N, H*W, C)
        feature_sequence = features.reshape(N, C, H * W).permute(0, 2, 1)
        
        # Add positional info
        positioned_features = feature_sequence + self.pos_embedding
        
        # Flatten to latent vector z
        z = positioned_features.reshape(N, -1)

        # --- DECODING ---
        decoded_sequence = self.decoder_input(z)
        decoded_sequence = decoded_sequence.reshape(N, self.sequence_length, self.channels)

        # Permute and reshape back to a spatial grid: (N, H*W, C) -> (N, C, H*W) -> (N, C, H, W)
        spatial_grid = decoded_sequence.permute(0, 2, 1).reshape(N, self.channels, self.spatial_size, self.spatial_size)
        
        reconstructed_image = self.decoder(spatial_grid)

        return reconstructed_image, z

# =================== Example Usage ===================
if __name__ == '__main__':
    # You can now easily control the model's complexity
    model = PositionalAutoencoder(base_channels=32) # Creates a model with 64 final channels
    # model_heavy = PositionalAutoencoder(base_channels=32) # Creates a model with 128 final channels
    
    input_images = torch.randn(4, 1, 128, 128)
    reconstructed_images, latent_vector = model(input_images)
    
    print("✅ Model updated successfully!")
    print(f"Original Image Shape:     {input_images.shape}")
    print(f"Latent Vector 'z' Shape:  {latent_vector.shape}")
    print(f"Reconstructed Image Shape:  {reconstructed_images.shape}")

    total_params_in_million = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total Trainable Parameters: {total_params_in_million:.2f} million")