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
from pipeline.models.autoencoderkl.autoencoder_kl import AutoencoderKL
from pipeline.helpers import modelcheckpointcallback, TrackGradNormCallback \
    , adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images, check_yaml, find_latest_ckpt
"""
384x384
rec = l1
L_adv = -mean(D(x^))
w_adapt = grad(rec)/grad(l_adv) #to balance both losses
gen_loss = rec + disc_factor * w_adapt * L_adv
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'    

class Loss(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=64, disc_weight=1.0, use_actnorm=False, perceptual_weight=1.0, kl_weight=1.0, logvar_init=0.0):
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

    def calculate_adaptive_weight(self, nll_loss, disc_loss, last_layer):
        nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        disc_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grad) / (torch.norm(disc_grad) + 1e-4)
        d_weight = self.disc_weight * d_weight
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, last_layer, split, global_step):
        batch_size = inputs.size(0)
        rec_loss = F.l1_loss(reconstructions, inputs, reduction="mean")

        if self.perceptual_weight > 0:
            inputs_rgb = inputs.repeat(1, 3, 1, 1)
            reconstructions_rgb = reconstructions.repeat(1, 3, 1, 1)
            perceptual_loss = self.perceptual_loss(reconstructions_rgb, inputs_rgb).mean()
            rec_loss = rec_loss + self.perceptual_weight * perceptual_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / batch_size

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

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        self.scaling_factor = 0.18125
        state_dict = torch.load("/home/vatsal/NWM/pretrained_sevirlr_vae_8x8x64_v1.pt")
        self.autoencoder.load_state_dict(state_dict, strict=True)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        # x: (B, T, C, H, W)
        # B, T, C, H, W = x.shape
        # out = []
        # for i in range(T):
        #     frame = x[:, i]  # (B, C, H, W)
        #     z = self.autoencoder.encode(frame).sample()
        #     out.append(z.unsqueeze(1))
        # return torch.cat(out, dim=1)
        out = self.autoencoder.encode(x).mode()
        return out

    @torch.no_grad()
    def decode(self, x):
        # x: (B, T, latent_C, H, W)
        # B, T, C, H, W = x.shape
        # out = []
        # for i in range(T):
        #     frame = x[:, i]
        #     dec = self.autoencoder.decode(frame)
        #     out.append(dec.unsqueeze(1))
        # return torch.cat(out, dim=1)
        out = self.autoencoder.decode(x)
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 16×16 → 8×8 → 4×4 → 2×2 → 1×1  (all with convolutions)
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # (B,128, 8, 8)
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # (B,256, 4, 4)
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # (B,512, 2, 2)
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(1024, 1024, 1, 1, 0)   # (B,1024, 1, 1)
        self.fc = nn.Linear(1024, latent_dim)             # (B, latent_dim)

    def forward(self, x):
        x = self.conv(x)        # (B,1024,1,1)
        x = self.conv_out(x)    # (B,1024,1,1)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # project latent → 1024×1×1
        self.fc = nn.Linear(latent_dim, 1024)

        # 1×1 → 2×2 → 4×4 → 8×8 → 16×16
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1),  # (B,1024,2,2)
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),   # (B,512,4,4)
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),    # (B,256,8,8)
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),    # (B,128,16,16)
            nn.SiLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(128, 64, 1, 1, 0)  # (B,64,16,16)

    def forward(self, z):
        x = self.fc(z)                     # (B,1024)
        x = x.view(x.size(0), 1024, 1, 1)  # (B,1024,1,1)
        x = self.deconv(x)                 # (B,128,16,16)
        x = self.conv_out(x)               # (B,64,16,16)
        return x

class ConvModel(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = Autoencoder(cfg.autoencoder)

        self.loss = Loss(cfg.lpips.disc_start, 
                        disc_num_layers=cfg.lpips.disc_num_layers, 
                        disc_in_channels=cfg.lpips.disc_in_channels, 
                        disc_weight=cfg.lpips.disc_weight, 
                        use_actnorm=cfg.lpips.use_actnorm,
                        perceptual_weight=cfg.lpips.perceptual_weight,
                        kl_weight=cfg.lpips.kl_weight,
                        logvar_init=cfg.lpips.logvar_init
                        )
        
        self.predictor = ConvModel(latent_dim=cfg.predictor.latent_dim)

        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps
        self.accumulate_grad_batches = cfg.trainer.accumulate_grad_batches
        self.automatic_optimization = False

    def forward(self, x):
        z, out = self.predictor(x)
        return out, z
    
    def get_last_layer(self):
        return self.predictor.decoder.conv_out.weight

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()       

        inp = batch['vil'] #[b, c, h, w]
        encoded_inp = self.autoencoder.encode(inp)  # (B, LC, LH, LW)
        encoded_pred, z = self(encoded_inp)

        self.toggle_optimizer(g_opt)
        aeloss, log_dict_ae = self.loss(encoded_inp, encoded_pred, z, optimizer_idx = 0, last_layer = self.get_last_layer(), split="train", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        aeloss = aeloss / self.accumulate_grad_batches

        self.manual_backward(aeloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(g_opt, gradient_clip_val=self.cfg.optim.gradient_clip_val)
            g_opt.step()
            g_sch.step()
            g_opt.zero_grad(set_to_none=True)
        self.untoggle_optimizer(g_opt)

        #discriminator step
        if self.global_step >= self.cfg.lpips.disc_start:
            self.toggle_optimizer(d_opt)
            discloss, log_dict_disc = self.loss(encoded_inp, encoded_pred, z, optimizer_idx = 1, last_layer = self.get_last_layer(), split="train", global_step=self.global_step)
            self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
            discloss = discloss / self.accumulate_grad_batches

            self.manual_backward(discloss)
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.clip_gradients(d_opt, gradient_clip_val=self.cfg.optim.gradient_clip_val)
                d_opt.step()
                d_sch.step()
                d_opt.zero_grad(set_to_none=True)
            self.untoggle_optimizer(d_opt)
            
        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            decoded_pred = self.autoencoder.decode(encoded_pred)
            log_metrics(decoded_pred.unsqueeze(2), inp.unsqueeze(2), "train", self)
            log_wandb_images(decoded_pred, inp, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
    
    def validation_step(self, batch, batch_idx):
        inp = batch['vil'] #[b, c, h, w]
        encoded_inp = self.autoencoder.encode(inp)  # (B, LC, LH, LW)
        encoded_pred, z = self(encoded_inp)

        aeloss, log_dict_ae = self.loss(encoded_inp, encoded_pred, z, optimizer_idx = 0, last_layer = self.get_last_layer(), split="val", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        discloss, log_dict_disc = self.loss(encoded_inp, encoded_pred, z, optimizer_idx = 1, last_layer = self.get_last_layer(), split="val", global_step=self.global_step)
        self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
        
        decoded_pred = self.autoencoder.decode(encoded_pred)
        log_metrics(decoded_pred.unsqueeze(2), inp.unsqueeze(2), "val", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, inp, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        
    def configure_optimizers(self):
        opt_ae = adamw_optimizer(self.autoencoder, self.cfg.optim.lr, self.cfg.optim.weight_decay, 
                                 beta1=self.cfg.optim.beta1, beta2=self.cfg.optim.beta2)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch_ae = cosine_warmup_scheduler(opt_ae, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)

        opt_disc = adamw_optimizer(self.loss.discriminator, self.cfg.lpips.disc_peak_lr, self.cfg.optim.weight_decay, 
                                   beta1=self.cfg.lpips.disc_beta1, beta2=self.cfg.lpips.disc_beta2)
        sch_params = self.cfg.lpips
        warmup_steps = sch_params.disc_warmup_ratio * self.total_steps
        sch_disc = cosine_warmup_scheduler(opt_disc, sch_params.disc_start_lr, sch_params.disc_final_lr, sch_params.disc_peak_lr, self.total_steps, warmup_steps)
        return [
                {"optimizer": opt_ae, "lr_scheduler": {"scheduler": sch_ae, "interval": "step", "frequency": 1}},
                {"optimizer": opt_disc, "lr_scheduler": {"scheduler": sch_disc, "interval": "step", "frequency": 1}}
            ]
    
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
