import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pipeline.helpers import log_gradients_paramater, modelcheckpointcallback, \
    adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images
"""
384x384
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

class MLP(nn.Module):
    """
    B, 5 -> B, 8
    """
    def __init__(self, inp_seq_len = 5, out_var_len = 8, hidden_dim = 128):
        super().__init__()
        self.inp_seq_len = inp_seq_len
        self.out_var_len = out_var_len 
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(inp_seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_var_len),
        )
    
    def forward(self, x):
        return self.mlp(x)

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = MLP()
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps
        self.model = torch.compile(self.model)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2).unsqueeze(2) #[b, t c, h, w]
        inp, target = inp[:, :self.input_frames], inp[:, self.input_frames:]
        b, t, c, h, w = inp.shape
        
        inp_intensities = inp.reshape(b, t, -1).mean(dim = 2)
        target_intensities_mean = target.reshape(b, t, -1).reshape(b, 4, t//4, -1).mean(dim=[2,3])
        target_intensities_var = target.reshape(b, t, -1).reshape(b, 4, t//4, -1).std(dim=[2,3])
        target = torch.cat([target_intensities_mean, target_intensities_var], dim = -1)
        pred_intensities = self(inp_intensities)
        loss = self.criterion(pred_intensities, target)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2).unsqueeze(2) #[b, t c, h, w]
        inp, target = inp[:, :self.input_frames], inp[:, self.input_frames:]
        b, t, c, h, w = inp.shape
        
        inp_intensities = inp.reshape(b, t, -1).mean(dim = 2)
        target_intensities_mean = target.reshape(b, t, -1).reshape(b, 4, t//4, -1).mean(dim=[2,3])
        target_intensities_var = target.reshape(b, t, -1).reshape(b, 4, t//4, -1).std(dim=[2,3])
        target = torch.cat([target_intensities_mean, target_intensities_var], dim = -1)
        pred_intensities = self(inp_intensities)
        loss = self.criterion(pred_intensities, target)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = adamw_optimizer(self.model, self.cfg.optim.lr, self.cfg.optim.weight_decay)
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
        print(colored(f"Resuming from checkpoint: {resume_ckpt} with run_id: {run_id}", "green"))

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

    trainer.fit(model, dm, ckpt_path=resume_ckpt)
    print("done")