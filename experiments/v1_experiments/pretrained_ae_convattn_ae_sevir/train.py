import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.models.autoencoderkl.autoencoder_kl import AutoencoderKL
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pipeline.helpers import load_checkpoint_cascast, log_gradients_paramater, modelcheckpointcallback, \
    adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images, log_wandb_images
"""
384x384
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        self.scaling_factor = 0.18125
        load_checkpoint_cascast("/home/vatsal/NWM/PreDiff/scripts/vae/sevirlr/autoencoder_ckpt.pth", self.autoencoder)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]  # (B, C, H, W)
            z = self.autoencoder.encode(frame).sample()
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)
        # out = self.autoencoder.encode(x.reshape(B*T, C, H, W)).mode()
        # return out.reshape(B, T, 4, 48, 48)

    @torch.no_grad()
    def decode(self, x):
        # x: (B, T, latent_C, H, W)
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]
            dec = self.autoencoder.decode(frame)
            out.append(dec.unsqueeze(1))
        return torch.cat(out, dim=1)
        # out = self.autoencoder.decode(x.reshape(B*T, C, H, W))
        # return out.reshape(B, T, 1, 384, 384)

class ConvAttnModel(nn.Module):
    """
    Main reason is since we are flatenning and loosing a lot of spatial information this 
    adds positional embedding and attention to attend to the spatial information.
    so after adding 2d pos embedding we process it 
    """
    def __init__(self, in_channels = 4, transformer_embed_dim = 128, nhead = 8, num_tf_layers = 4, latent_dim = 512):
        super().__init__()
        self.transformer_embed_dim = transformer_embed_dim
        #48, 48 -> tf_embed_dim, 12, 12
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(), #empirical since using tf below.
            nn.Conv2d(64, transformer_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, transformer_embed_dim),
            nn.GELU(),
        )

        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, 144, transformer_embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_embed_dim, 
                                                    nhead=nhead,
                                                    dim_feedforward=transformer_embed_dim*4,
                                                    activation = 'gelu',
                                                    batch_first = True,
                                                    norm_first = True,
                                                    )
        self.encoder_tf = nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers)
        
        #instead of avg pooling we use attention (weighted pooling)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, transformer_embed_dim))
        self.attention_pool = nn.MultiheadAttention(embed_dim=transformer_embed_dim, num_heads=nhead, batch_first=True)

        self.encoder_head = nn.Sequential(
            nn.LayerNorm(transformer_embed_dim),
            nn.Linear(transformer_embed_dim, latent_dim),
        )
        #-------decoder-------
        self.decoder_head = nn.Linear(latent_dim, transformer_embed_dim)
        self.decoder_queries = nn.Parameter(torch.randn(1, 144, transformer_embed_dim))

        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, 144, transformer_embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_embed_dim, 
                                                    nhead=nhead,
                                                    dim_feedforward=transformer_embed_dim*4,
                                                    activation = 'gelu',
                                                    batch_first = True,
                                                    norm_first = True,
                                                    )
        self.decoder_tf = nn.TransformerDecoder(decoder_layer, num_layers=num_tf_layers)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(transformer_embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        input : b 4 48 48 
        """
        b, c, h, w = x.shape
        x = self.encoder_cnn(x) #[b. 128. 12, 12]
        x = x.flatten(2).transpose(1, 2) #[b, 144, 128]
        x = x + self.encoder_pos_embedding
        context = self.encoder_tf(x) #[b, 144, 128]
        query = self.pooling_query.expand(b, -1, -1) #[b, 1, 128]
        pooled, _ = self.attention_pool(query, context, context) #[b, 1, 128]
        latent = self.encoder_head(pooled) #[b, 1, 512]
        return latent.squeeze(1) #[b, 512]
    
    def decode(self, z):
        """
        input : b latent_dim[512]
        """
        b = z.shape[0]
        context = self.decoder_head(z).unsqueeze(1) #[b, 1, 128]
        query = self.decoder_queries.expand(b, -1, -1) #[b, 144, 128]
        query = query + self.decoder_pos_embedding
        patches = self.decoder_tf(query, context) #[b, 144, 128]
        x = patches.transpose(1, 2).reshape(b, self.transformer_embed_dim, 12, 12) #[b, 128, 12, 12]
        x = self.decoder_cnn(x)
        return x
    
    def forward(self, x):
        """
        input : b 4 48 48 
        """
        z = self.encode(x)
        out = self.decode(z)
        return z, out

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = Autoencoder(cfg.autoencoder)
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps
        self.predictor = ConvAttnModel()
        self.predictor = torch.compile(self.predictor)
        self.criterion = nn.HuberLoss()

    def forward(self, x):
        """
        input b t 4 48 48 
        """
        assert x.shape[1] == 1, "input should be 1 frame"
        x = x.squeeze(1)
        z, out = self.predictor(x)
        out = out.unsqueeze(1)
        return out

    def training_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2).unsqueeze(2)
        encoded_inp = self.autoencoder.encode(inp)  # (B, T, LC, LH, LW)
        b, t, c, h, w = encoded_inp.shape
        
        encoded_pred = self(encoded_inp)
        loss = self.criterion(encoded_pred, encoded_inp)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        if batch_idx % log_interval == 0:
            decoded_pred = self.autoencoder.decode(encoded_pred)
            log_metrics(decoded_pred, inp, "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, inp, f"Train_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2).unsqueeze(2)
        encoded_inp = self.autoencoder.encode(inp)  # (B, T, LC, LH, LW)
        b, t, c, h, w = encoded_inp.shape
        
        encoded_pred = self(encoded_inp)
        loss = self.criterion(encoded_pred, encoded_inp)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        log_interval = int(self.cfg.logging.log_val_all_metrics_n * self.cfg.trainer.total_val_steps)
        if batch_idx % log_interval == 0:
            decoded_pred = self.autoencoder.decode(encoded_pred)
            log_metrics(decoded_pred, inp, "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, inp, f"Val_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss

    def configure_optimizers(self):
        opt = adamw_optimizer(self.predictor, self.cfg.optim.lr, self.cfg.optim.weight_decay)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch = cosine_warmup_scheduler(opt, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    resume_ckpt = args.resume
    run_id = args.run_id

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok=True)

    dm = SEVIRLightningDataModule(cfg.dataset); dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for data in loader:
            print(f"Data shape: {data.shape}")
            break

    total_train_steps = (len(dm.train_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_val_steps = (len(dm.val_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_test_steps = (len(dm.test_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    cfg.trainer.total_train_steps = int(total_train_steps)
    cfg.trainer.total_val_steps = int(total_val_steps)
    cfg.trainer.total_test_steps = int(total_test_steps)
    if cfg.trainer.limit_train_batches is not None:
        cfg.trainer.total_train_steps = total_train_steps * cfg.trainer.limit_train_batches
    if cfg.trainer.limit_val_batches is not None:
        cfg.trainer.total_val_steps = total_val_steps * cfg.trainer.limit_val_batches
    if cfg.trainer.limit_test_batches is not None:
        cfg.trainer.total_test_steps = total_test_steps * cfg.trainer.limit_test_batches

    logger = WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = os.path.join(cfg.experiment_path, 'outputs'), id = run_id)
    run_id = logger.experiment.id
    run_dir = logger.experiment.dir
    artifact = wandb.Artifact(cfg.experiment_name, type="code")
    artifact.add_file(os.path.join(os.path.dirname(__file__), "train.py"))
    logger.experiment.log_artifact(artifact)

    checkpoint_callback = modelcheckpointcallback(run_dir, cfg.trainer.total_train_steps, cfg.trainer.save_every_n_steps, cfg.trainer.save_on_train_epoch_end)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs, 
        accelerator='gpu', 
        devices=cfg.trainer.devices,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        track_grad_norm = 2,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    model = Model(cfg)
    log_gradients_paramater(model, cfg.trainer.total_train_steps, cfg.logging.wandb_watch_log_freq, logger)

    trainer.fit(model, dm)
    print("done")