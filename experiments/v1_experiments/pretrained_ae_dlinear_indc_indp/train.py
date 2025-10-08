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

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len * 4,self.pred_len * 4))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/(4*self.seq_len))*torch.ones([self.pred_len * 4,self.seq_len * 4]))
                self.Linear_Trend.append(nn.Linear(self.seq_len * 4,self.pred_len * 4))
                self.Linear_Trend[i].weight = nn.Parameter((1/(4*self.seq_len))*torch.ones([self.pred_len * 4, self.seq_len * 4]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len * 4,self.pred_len*4)
            self.Linear_Trend = nn.Linear(self.seq_len*4,self.pred_len*4)
            self.Linear_Seasonal.weight = nn.Parameter((1/(4*self.seq_len))*torch.ones([self.pred_len*4,self.seq_len*4]))
            self.Linear_Trend.weight = nn.Parameter((1/(4*self.seq_len))*torch.ones([self.pred_len*4,self.seq_len*4]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len * 4],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len * 4],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

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
        self.predictor = DLinear(cfg.dlinear)

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

        pred = self(inp.reshape(b, self.input_frames * c, h * w)).reshape(b, self.pred_frames, c, h, w)
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
        pred = self(inp.reshape(b, self.input_frames * c,  h * w))
        pred = pred.reshape(b, self.pred_frames, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        log_metrics(decoded_pred, decoded_tgt, "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss

    def test_step(self, batch, batch_idx):
        v = batch.permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t
        pred = self(inp.reshape(b, self.input_frames * c,  h * w))
        pred = pred.reshape(b, self.pred_frames, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        log_metrics(decoded_pred, decoded_tgt, "test", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Test_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss
        
    def configure_optimizers(self):
        opt = adamw_optimizer(self.predictor, self.cfg.optim.lr, self.cfg.optim.weight_decay)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch = cosine_warmup_scheduler(opt, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    resume_ckpt = args.resume
    run_id = args.run_id
    if run_id is not None:
        print(colored(f"Resuming from checkpoint: {resume_ckpt} with run_id: {run_id}", "green"))

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = OmegaConf.load(config_path)
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

    logger = WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = os.path.join(cfg.experiment_path, 'outputs'), id = run_id, resume = "allow")
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
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    model = Model(cfg)
    # log_gradients_paramater(model, cfg.trainer.total_train_steps, cfg.logging.wandb_watch_log_freq, logger)

    trainer.fit(model, dm, ckpt_path=resume_ckpt)
    print("done")