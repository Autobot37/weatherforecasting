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
from pipeline.helpers import check_yaml, TrackGradNormCallback, modelcheckpointcallback
"""
384x384
"""
os.environ['WANDB_API_KEY'] = 'wandb key'

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.autoencoder = AutoencoderKL(**cfg.autoencoder)
        self.cfg = cfg

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        #-------- calulating loss -----------------
        inp = batch.permute(0,3,1,2).unsqueeze(2)
        encoded_inp = self.autoencoder.encode(inp)  # (B, T, LC, LH, LW)
        b, t, c, h, w = encoded_inp.shape
        
        encoded_pred = self(encoded_inp)
        loss = self.criterion(encoded_pred, encoded_inp)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) # logs loss every step and logs separately averaging over epoch.
        #------------------------------------------

        #-------- logging --------------------------
        # logs metrics and images at intervals when global step(all batches in all epochs) is multiple of interval
        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        global_step = self.trainer.global_step

        if global_step % log_interval == 0:
            decoded_pred = self.autoencoder.decode(encoded_pred)
            log_metrics(decoded_pred, inp, "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if global_step % plot_interval == 0:
            log_wandb_images(decoded_pred, inp, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        #------------------------------------------
        return loss
    
    def validation_step(self, batch, batch_idx):
        ###-------- calulating loss -----------------
        inp = batch.permute(0,3,1,2).unsqueeze(2)
        encoded_inp = self.autoencoder.encode(inp)  # (B, T, LC, LH, LW)
        b, t, c, h, w = encoded_inp.shape
        
        encoded_pred = self(encoded_inp)
        loss = self.criterion(encoded_pred, encoded_inp)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) # logs loss every step and logs separately averaging over epoch.
        #------------------------------------------

        ###-------- logging --------------------------
        # logs metrics and images at intervals when global step(all batches in all epochs) is multiple of interval

        global_step = self.trainer.global_step
        log_interval = int(self.cfg.logging.log_val_all_metrics_n * self.cfg.trainer.total_val_steps)
        if global_step % log_interval == 0:
            decoded_pred = self.autoencoder.decode(encoded_pred)
            log_metrics(decoded_pred, inp, "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if global_step % plot_interval == 0:
            log_wandb_images(decoded_pred, inp, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        #------------------------------------------
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
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from checkpoint")
    args, unknown = parser.parse_known_args()

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = OmegaConf.load(config_path)

    ## check valid fields from cmd line or .sh file and override config.yaml
    cli_cfg = OmegaConf.from_dotlist(unknown)
    check_yaml(cfg, cli_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    ## if resume is true, find the latest checkpoint automatically else start fresh
    if args.resume:
        try:
            wandb_dir = os.path.join(cfg.experiment_path, 'outputs', cfg.experiment_name, 'wandb')
            if not os.path.exists(wandb_dir):
                raise FileNotFoundError(f"Wandb directory {wandb_dir} does not exist")
            
            run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-") and os.path.isdir(os.path.join(wandb_dir, d))]
            if not run_dirs:
                raise ValueError("No run directories found in wandb_dir")
            
            latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d)))
            latest_run_path = os.path.join(wandb_dir, latest_run_dir)
            ckpt_dir = os.path.join(latest_run_path, 'checkpoints')
            
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist")
            
            ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt"))
            if not ckpts:
                raise FileNotFoundError("No checkpoint files found")
            
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            run_id = latest_run_dir.split('-')[-1]  
            print(colored(f"Resuming from latest checkpoint: {ckpt_path} from directory {latest_run_dir} with run id {run_id}", "green"))
            
        except (FileNotFoundError, ValueError, IndexError, OSError) as e:
            print(colored(f"Cannot resume: {e}", "yellow"))
            print(colored("Starting fresh training...", "cyan"))
            args.resume = False

    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok=True)

    dm = SEVIRLightningDataModule(**cfg.dataset)
    dm.setup()
    dm.prepare_data()    
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for data in loader:
            print(f"Data shape: {data['vil'].shape}")
            break

    ## total steps = (num_batches in loader * num_epochs) / accumulate_grad_batches
    total_train_steps = (len(dm.train_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_val_steps = (len(dm.val_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_test_steps = (len(dm.test_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    cfg.trainer.total_train_steps = int(total_train_steps)
    cfg.trainer.total_val_steps = int(total_val_steps)
    cfg.trainer.total_test_steps = int(total_test_steps)

    ## if limit batches is set, then override total steps with fraction, it simply breaks the dataloader iteration after limit.
    ## Data shuffling depends on dataloader shuffle parameter not here. So if shuffle is True, it will return random sequence of batches every epoch.
    ## else if shuffle is False, it will return same sequence of batches every epoch.
    if cfg.trainer.limit_train_batches is not None:
        cfg.trainer.total_train_steps = total_train_steps * cfg.trainer.limit_train_batches
    if cfg.trainer.limit_val_batches is not None:
        cfg.trainer.total_val_steps = total_val_steps * cfg.trainer.limit_val_batches
    if cfg.trainer.limit_test_batches is not None:
        cfg.trainer.total_test_steps = total_test_steps * cfg.trainer.limit_test_batches

    ## wandb config and directories setup 
    exp_name = cfg.experiment_name
    save_dir = os.path.join(cfg.experiment_path, 'outputs', exp_name)
    logger = WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = save_dir, resume = "allow", id = run_id if args.resume else None)
    run_id = logger.experiment.id
    run_dir = logger.experiment.dir

    ## config.yaml is uploaded to wandb but upload train.py too for reference
    artifact = wandb.Artifact(cfg.experiment_name, type="code")
    artifact.add_file(os.path.join(os.path.dirname(__file__), "train.py"))
    logger.experiment.log_artifact(artifact)

    ## there are three callbacks used
    # 1. ModelCheckpoint[@pipeline/helpers.py] - currently it is configured to save all checkpoints at some interval = int(total_train_steps * cfg.trainer.save_every_n_steps) and at the end of every epoch.
    #    but it can be changed to save only top k checkpoints based on some metric like val_loss.
    # 2. LearningRateMonitor - logs learning rate at each step
    # 3. TrackGradNormCallback[@pipeline/helpers.py] - logs gradient norm of model at each step [single scalar value.]

    checkpoint_callback = modelcheckpointcallback(run_dir, cfg.trainer.total_train_steps, cfg.trainer.save_every_n_steps, cfg.trainer.save_on_train_epoch_end)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs, 
        accelerator='gpu', 
        devices=cfg.trainer.devices,
        strategy="auto",
        callbacks=[checkpoint_callback, lr_monitor_callback, TrackGradNormCallback()],
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    model = Model(cfg)
    trainer.fit(model, dm, ckpt_path=ckpt_path if args.resume else None)
    ## success marker to indicate successful completion of training
    print("done")
