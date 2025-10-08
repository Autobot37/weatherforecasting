import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.models.autoencoderkl.losses.contperceptual import adopt_weight, hinge_d_loss, NLayerDiscriminator, weights_init, LPIPS
from pipeline.datasets.sevire.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pipeline.models.autoencoderkl.custom_akl import AutoencoderKL
from pipeline.helpers import load_checkpoint_cascast, log_gradients_paramater, modelcheckpointcallback, TrackGradNormCallback \
    , adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images, check_yaml, find_latest_ckpt
from pytorch_msssim import ssim
from pipeline.models.ae_64x8x8_tf import PosAwareAE_TF
"""
384x384
rec = l1
L_adv = -mean(D(x^))
w_adapt = grad(rec)/grad(l_adv) #to balance both losses
gen_loss = rec + disc_factor * w_adapt * L_adv
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'    
class Loss(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, use_actnorm=False, perceptual_weight=1.0, kl_weight=1.0, logvar_init=0.0, recon_weight = 1.0):
        super().__init__()
        self.disc_start = disc_start
        self.disc_weight = disc_weight

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight

    def calculate_adaptive_weight(self, nll_loss, disc_loss, last_layer):
        nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        disc_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grad) / (torch.norm(disc_grad) + 1e-4)
        d_weight = self.disc_weight * d_weight
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, last_layer, split, global_step):
        rec_loss = self.recon_weight * F.l1_loss(reconstructions, inputs, reduction="mean")

        if self.perceptual_weight > 0:
            inputs_rgb = inputs.repeat(1, 3, 1, 1)
            reconstructions_rgb = reconstructions.repeat(1, 3, 1, 1)
            # perceptual_loss = self.perceptual_loss(reconstructions_rgb, inputs_rgb).mean()
            # rec_loss = rec_loss + self.perceptual_weight * perceptual_loss
            perceptual_loss = 1 - ssim(inputs_rgb, reconstructions_rgb, data_range=1.0)
            rec_loss = rec_loss + self.perceptual_weight * perceptual_loss

        nll_loss = rec_loss

        if global_step < self.disc_start:
            return nll_loss, {
            f"{split}/total_loss": nll_loss.detach().cpu().mean(),
            f"{split}/rec_loss": rec_loss.detach().cpu().mean(),
            f"{split}/nll_loss": nll_loss.detach().cpu().mean(),
            f"{split}/g_loss": torch.tensor(0.0).cpu(),
            f"{split}/d_weight": torch.tensor(0.0).cpu(),
            }

        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = nll_loss + d_weight * g_loss
            return loss, {
            f"{split}/total_loss": loss.detach().cpu().mean(),
            f"{split}/nll_loss": nll_loss.detach().cpu().mean(),
            f"{split}/rec_loss": rec_loss.detach().cpu().mean(),
            f"{split}/g_loss": g_loss.detach().cpu().mean(),
            f"{split}/d_weight": d_weight.detach().cpu().mean(),
            }

        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            d_loss = hinge_d_loss(logits_real, logits_fake)
            return d_loss, {
            f"{split}/disc_loss": d_loss.detach().cpu().mean(),
            f"{split}/logits_real": logits_real.detach().mean().cpu(),
            f"{split}/logits_fake": logits_fake.detach().mean().cpu(),
            }

class Losskl(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, use_actnorm=False, perceptual_weight=1.0, kl_weight=1.0, logvar_init=0.0, recon_weight = 1.0):
        super().__init__()
        self.disc_start = disc_start
        self.disc_weight = disc_weight

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight

    def calculate_adaptive_weight(self, nll_loss, disc_loss, last_layer):
        nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        disc_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grad) / (torch.norm(disc_grad) + 1e-4)
        d_weight = self.disc_weight * d_weight
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, last_layer, split, global_step):
        batch_size = inputs.size(0)
        rec_loss = self.recon_weight * F.l1_loss(reconstructions, inputs, reduction="mean")

        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # nll_loss = torch.sum(nll_loss) / batch_size
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / batch_size

        nll_loss = rec_loss + kl_loss * self.kl_weight

        if global_step < self.disc_start:
            return nll_loss, {
            f"{split}/total_loss": nll_loss.detach().cpu().mean(),
            f"{split}/rec_loss": rec_loss.detach().cpu().mean(),
            f"{split}/kl_loss": kl_loss.detach().cpu().mean(),
            f"{split}/nll_loss": nll_loss.detach().cpu().mean(),
            f"{split}/g_loss": torch.tensor(0.0).cpu(),
            f"{split}/d_weight": torch.tensor(0.0).cpu(),
            }

        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = nll_loss + d_weight * g_loss
            return loss, {
            f"{split}/total_loss": loss.detach().cpu().mean(),
            f"{split}/nll_loss": nll_loss.detach().cpu().mean(),
            f"{split}/kl_loss": kl_loss.detach().cpu().mean(),
            f"{split}/rec_loss": rec_loss.detach().cpu().mean(),
            f"{split}/g_loss": g_loss.detach().cpu().mean(),
            f"{split}/d_weight": d_weight.detach().cpu().mean(),
            }

        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            d_loss = hinge_d_loss(logits_real, logits_fake)
            return d_loss, {
            f"{split}/disc_loss": d_loss.detach().cpu().mean(),
            f"{split}/logits_real": logits_real.detach().mean().cpu(),
            f"{split}/logits_fake": logits_fake.detach().mean().cpu(),
            }

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = PosAwareAE_TF()

        self.loss = Loss(cfg.lpips.disc_start, 
                        disc_num_layers=cfg.lpips.disc_num_layers, 
                        disc_in_channels=cfg.lpips.disc_in_channels, 
                        disc_weight=cfg.lpips.disc_weight, 
                        use_actnorm=cfg.lpips.use_actnorm,
                        perceptual_weight=cfg.lpips.perceptual_weight,
                        kl_weight=cfg.lpips.kl_weight,
                        logvar_init=cfg.lpips.logvar_init,
                        recon_weight=cfg.lpips.recon_weight
                        )
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps

    def forward(self, x):
        recon, z = self.autoencoder(x)
        return recon, z
    
    def get_last_layer(self):
        return self.autoencoder.dec[-1].weight

    def training_step(self, batch, batch_idx):     
        inp = batch['vil'] #[b, c, h, w]
        pred, z = self(inp)

        aeloss, log_dict_ae = self.loss(inp, pred, z, optimizer_idx = 0, last_layer = self.get_last_layer(), split="train", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)

        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        if batch_idx % log_interval == 0:
            log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if (self.current_epoch * self.cfg.trainer.total_train_steps + batch_idx) % plot_interval == 0:
            log_wandb_images(pred, inp, f"Train Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return aeloss
            
    def validation_step(self, batch, batch_idx):
        inp = batch['vil'] #[b, c, h, w]
        pred, z = self(inp)

        aeloss, log_dict_ae = self.loss(inp, pred, z, optimizer_idx = 0, last_layer = self.get_last_layer(), split="val", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        
        log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "val", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_val_steps)
        if (self.current_epoch * self.cfg.trainer.total_val_steps + batch_idx) % plot_interval == 0:
            log_wandb_images(pred, inp, f"Val_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return aeloss

    def test_step(self, batch, batch_idx):
        inp = batch['vil'] #[b, c, h, w]
        pred, z = self(inp)

        aeloss, log_dict_ae = self.loss(inp, pred, z, optimizer_idx = 0, last_layer = self.get_last_layer(), split="test", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        discloss, log_dict_disc = self.loss(inp, pred, z, optimizer_idx = 1, last_layer = self.get_last_layer(), split="test", global_step=self.global_step)
        self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
        
        log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "test", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_test_steps)
        if (self.current_epoch * self.cfg.trainer.total_test_steps + batch_idx) % plot_interval == 0:
            log_wandb_images(pred, inp, f"Test_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}_test", self)
        
    def configure_optimizers(self):
        opt_ae = adamw_optimizer(self.autoencoder, self.cfg.optim.lr, self.cfg.optim.weight_decay, 
                                 beta1=self.cfg.optim.beta1, beta2=self.cfg.optim.beta2)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch_ae = cosine_warmup_scheduler(opt_ae, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)

        return {"optimizer": opt_ae, "lr_scheduler": {"scheduler": sch_ae, "interval": "step", "frequency": 1}}
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from checkpoint")
    args, unknown = parser.parse_known_args()

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = OmegaConf.load(config_path)

    cli_cfg = OmegaConf.from_dotlist(unknown)
    check_yaml(cfg, cli_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if args.resume:
        ckpt_path, run_id = find_latest_ckpt(cfg)
        if ckpt_path is None:
            args.resume = False
            print(colored("No checkpoint found, starting from scratch.", "yellow"))
        else:
            print(colored(f"Resuming from checkpoint: {ckpt_path} with run id {run_id}", "green"))

    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok=True)

    dm = SEVIRLightningDataModule(
        dataset_name=cfg.dataset.name,
        num_workers=cfg.dataset.num_workers,
        batch_size=cfg.dataset.batch_size,
        seq_len=cfg.dataset.seq_len,
        stride=cfg.dataset.stride,
        layout='NTHW'
    )
    dm.setup()
    dm.prepare_data()    
    for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        print(colored(f"Number of batches in dataloader: {len(loader)}", "cyan"))
        for data in loader:
            print(f"Data shape: {data['vil'].shape}")
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
    cfg.lpips.disc_start = int(cfg.lpips.disc_start * cfg.trainer.total_train_steps)

    exp_name = cfg.experiment_name
    save_dir = os.path.join(cfg.experiment_path, 'outputs', exp_name)
    logger = WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = save_dir, resume = "allow", id = run_id if args.resume else None)
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
    print("done")