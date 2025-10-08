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
from pipeline.helpers import load_checkpoint_cascast, log_gradients_paramater, modelcheckpointcallback \
    , adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images
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

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = Autoencoder(cfg.autoencoder)
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps
        self.predictor = nn.Linear(self.input_frames * 4, self.pred_frames * 4)

    def forward(self, x):
        out = self.predictor(x)
        return out

    def training_step(self, batch, batch_idx):
        v = batch.permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)  # (B, T, LC, LH, LW)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t

        pred = self(inp.permute(0, 3, 4, 1, 2).reshape(b, h, w, 13*c)).permute(0, 3, 1, 2).reshape(b, 12, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        if batch_idx % log_interval == 0:
            pred = pred + inp_t
            tgt = tgt + inp_t

            decoded_pred = self.autoencoder.decode(pred)
            decoded_tgt = self.autoencoder.decode(tgt)
            log_metrics(decoded_pred, decoded_tgt, "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss
    
    def validation_step(self, batch, batch_idx):
        v = batch.permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t
        pred = self(inp.permute(0, 3, 4, 1, 2).reshape(b, h, w, 13*c)).permute(0, 3, 1, 2).reshape(b, 12, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        log_interval = int(self.cfg.logging.log_val_all_metrics_n * self.cfg.trainer.total_val_steps)
        if batch_idx % log_interval == 0:
            log_metrics(decoded_pred, decoded_tgt, "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
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
    args = parser.parse_args()
    resume_ckpt = args.resume
    run_id = None
    if resume_ckpt is not None:
        run_id = resume_ckpt.split("/")[-4].split("-")[-1]

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    cfg = OmegaConf.load("experiments/pretrained_ae_linear_sevir/config.yaml")
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

    trainer.fit(model, dm, ckpt_path=resume_ckpt)
    print("done")