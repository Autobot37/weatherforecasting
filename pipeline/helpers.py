import torch
from collections import OrderedDict
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import os
from pipeline.metrics import calc_metrics
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
from termcolor import colored
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pipeline.datasets.sevir.sevir import vil_cmap

def load_checkpoint_cascast(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    ckpt_submodels = list(checkpoint_model.keys())
    submodels = ['autoencoder_kl']
    key = 'autoencoder_kl'
    if key not in submodels:
        print(f"warning!!!!!!!!!!!!!: skip load of {key}")
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model[key].items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("net."):
            name = name[len("net."):]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    print(colored(f"loaded {key} successfully the game is on", 'green'))
    return model

def lr_range_test(model, optimizer, train_dataloader, val_dataloader, criterion, input_frames, experiment_path, device, max_lr, num_iter):
    """
    generally /10 were lr explodes gradients.
    """
    class CustomTrainIter(TrainDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data):
            #batch data = [b, h, w, t]
            data = batch_data.permute(0, 3, 1, 2)
            inp, target = data[:, :input_frames], data[:, input_frames:]
            return inp, target

    class CustomValIter(ValDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data):
            #batch data = [b, h, w, t]
            data = batch_data.permute(0, 3, 1, 2)
            inp, target = data[:, :input_frames], data[:, input_frames:]
            return inp, target

    custom_train_dataloader = CustomTrainIter(train_dataloader)
    custom_val_dataloader = CustomValIter(val_dataloader)
    outputs_path = os.path.join(experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok = True)

    lr_finder = LRFinder(model, optimizer, criterion, device = device)
    lr_finder.range_test(custom_train_dataloader, val_loader = custom_val_dataloader, end_lr = max_lr, num_iter = num_iter)
    fig = lr_finder.plot()
    fig.savefig(os.path.join(outputs_path, 'lr_range_test.png'))
    lr_finder.reset()

def adamw_optimizer(model, lr, weight_decay, beta1=0.9, beta2=0.999):
    """
    https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
    weight_decay in model architecture testing should be 0 since 
    we are testing the mdoels ability to learn.
    otherwise generally it should be 1e-2 for 50m models.
    it is a regularizater and need tuning also helps in finding simple intepretation if overparametrized.
    if using another regularization like dropout and data augmentations then still needed 
    since dropout forces more robustness and data augmentations forces invariance.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, 
                             betas=(beta1, beta2))

def cosine_warmup_scheduler(opt, start_lr, final_lr, peak_lr, total_steps, warmup_steps):
    """
    start->base_lr[in warmup]
    base_lr->final_lr[in rest steps] in single 2nd half of cosine cycle.
    general_values for scratch
    warmup_steps generally 10%.
    peak_lr generally 1e-3
    final_lr generally 1e-6
    start_lr generally 1e-4
    """
    for param_group in opt.param_groups:
        if param_group['lr'] != peak_lr:
            print(colored(f"lr is not peak lr, it is {param_group['lr']} changing to {peak_lr}", 'red'))
            param_group['lr'] = peak_lr

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor = start_lr / peak_lr,
        end_factor = 1.0,
        total_iters = warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max = total_steps - warmup_steps,
        eta_min = final_lr
    )
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers = [warmup_scheduler, cosine_scheduler],
        milestones = [warmup_steps]
    )
    return sch

def one_cycle_scheduler(opt, start_lr, peak_lr, final_lr, total_steps, rampup_steps):
    """
    start_lr -> peak_lr[in warmup steps]
    peak_lr -> final_lr[in rest steps]
    general_values
    peak_lr generally 1e-3
    start_lr generally peak_lr / 25 ~ 4e-5
    final_lr generally start_lr / 100 ~ 4e-7
    rampup_steps = 30% of total steps

    rampup steps should be higher like 1/4 of total steps.
    """
    if rampup_steps / total_steps < 0.2:
        print(colored(f"rampup steps should be higher than 20% of total steps, it is {rampup_steps / total_steps}", 'red'))

    # rampup percentage
    pct_start = rampup_steps / total_steps

    div_factor = peak_lr / start_lr
    final_div_factor = start_lr / final_lr

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=peak_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy='cos'
    )

    return scheduler

def log_metrics(predictions, targets, tag, pl_module):
    """
    predictions: (B, T, C, H, W) [0, 1]
    targets: (B, T, C, H, W) [0, 1]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach()
    metrics = calc_metrics(predictions, targets)
    metrics = {f"{tag}_{k}": v for k, v in metrics.items()}
    pl_module.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)

def log_wandb_images(predicted, target, label, pl_module, batch_idxs=4):
    """
    input : predicted, target in B T H W or B T C H W (c==1) [0, 1]
    output: Log separate images for first `batch_idxs` samples
    """
    predicted = predicted.detach().cpu()
    target = target.detach().cpu()

    # Check range validity
    in_range = ((target >= 0) & (target <= 1)).sum().item()
    total_elems = target.numel()
    ratio = in_range / total_elems
    if ratio < 0.9:
        print(f"\033[91mtarget data not in [0,1] range: {ratio:.2%}\033[0m")

    # Remove channel dim if present
    if predicted.ndim == 5:
        assert predicted.shape[2] == 1, "Predicted must be (B,T,1,H,W)"
        predicted = predicted.squeeze(2)
    if target.ndim == 5:
        assert target.shape[2] == 1, "Target must be (B,T,1,H,W)"
        target = target.squeeze(2)

    # Clamp batch_idxs to available size
    batch_idxs = min(batch_idxs, predicted.shape[0])

    target_np = (target.clamp(0,1) * 255).numpy().astype('uint8')
    pred_np = (predicted.clamp(0,1) * 255).numpy().astype('uint8')
    diff_np = abs(target_np.astype(float) - pred_np.astype(float)).clip(0,255).astype('uint8')

    B, T, H, W = target_np.shape

    cmap, norm, _, _ = vil_cmap()

    # ---- log each batch as separate figure ----
    for b in range(batch_idxs):
        fig, axes = plt.subplots(3, T, figsize=(4*T, 12))
        if T == 1:
            axes = axes.reshape(3, 1)

        for t in range(T):
            # Original
            ax0 = axes[0, t]
            im0 = ax0.imshow(target_np[b, t], cmap=cmap, norm=norm)
            ax0.set_title(f'Batch {b}, Time {t}: Original')
            ax0.axis('off')
            plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            # Reconstruction
            ax1 = axes[1, t]
            im1 = ax1.imshow(pred_np[b, t], cmap=cmap, norm=norm)
            ax1.set_title(f'Batch {b}, Time {t}: Reconstruction')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Difference
            ax2 = axes[2, t]
            im2 = ax2.imshow(diff_np[b, t], cmap='Reds', vmin=0, vmax=255)
            ax2.set_title(f'Batch {b}, Time {t}: Abs Diff')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if isinstance(pl_module.logger, WandbLogger):
            wandb_image = wandb.Image(fig, caption=f"{label} (batch {b})")
            pl_module.logger.experiment.log({
                f"{label}": wandb_image,
                'global_step': pl_module.global_step
            })
        plt.close(fig)

def log_gradients_paramater(model, total_train_steps, wandb_watch_log_freq, logger):
    """
    one time call for logging gradients and parameters and model architecture
    """
    if isinstance(logger, WandbLogger):
        log_freq = int(total_train_steps * wandb_watch_log_freq)
        logger.watch(model, log = "all", log_freq = log_freq)
    else:
        print(colored("logger is not a WandbLogger, skipping gradient and parameter logging", 'red'))

def modelcheckpointcallback(run_dir, total_train_steps, save_every_n_steps, save_on_train_epoch_end):
    return ModelCheckpoint(
        dirpath = os.path.join(run_dir, 'checkpoints'),
        filename = "{epoch}-{step:06d}",
        every_n_train_steps = int(total_train_steps * save_every_n_steps),
        save_on_train_epoch_end = save_on_train_epoch_end,
        save_top_k=-1,  # save all models
        save_last= True,
    )

class TrackGradNormCallback(pl.Callback):
    def __init__(self, norm_type=2):
        super().__init__()
        self.norm_type = norm_type

    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({"grad_norm": total_norm, "global_step": trainer.global_step})

def check_yaml(cfg, cli_cfg, path=""):
    for k in cli_cfg:
        full_key = f"{path}.{k}" if path else k
        if k not in cfg:
            raise KeyError(f"Invalid override key: '{full_key}' not found in base config")
        if isinstance(cli_cfg[k], dict) and isinstance(cfg[k], dict):
            check_yaml(cfg[k], cli_cfg[k], full_key)

def find_latest_ckpt(cfg):
    wandb_dir = os.path.join(cfg.experiment_path, 'outputs', cfg.experiment_name, 'wandb')
    if not os.path.exists(wandb_dir):
        return None, None

    # Collect all checkpoint paths with their mtimes and run_ids
    ckpt_entries = []

    for run_dir in os.listdir(wandb_dir):
        run_path = os.path.join(wandb_dir, run_dir)
        ckpt_dir = os.path.join(run_path, 'files/checkpoints')

        if not (run_dir.startswith("run-") and os.path.isdir(ckpt_dir)):
            continue

        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".ckpt"):
                fpath = os.path.join(ckpt_dir, fname)
                mtime = os.path.getmtime(fpath)
                run_id = run_dir.split("-")[-1]
                ckpt_entries.append((mtime, fpath, run_id))

    # Sort by most recent first
    ckpt_entries.sort(reverse=True)

    # Try loading each checkpoint until one works
    for _, fpath, run_id in ckpt_entries:
        try:
            _ = torch.load(fpath, map_location='cpu')
            return fpath, run_id
        except Exception as e:
            continue  # Try next

    return None, None