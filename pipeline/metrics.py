import torch
import numpy as np
import einops
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
#review
_eps = 1e-8

def _hit_miss_fa_cn(pred, target, threshold):
    p = (pred >= threshold).float()
    t = (target >= threshold).float()
    tp = torch.sum(p * t)
    fn = torch.sum((1 - p) * t)
    fp = torch.sum(p * (1 - t))
    tn = torch.sum((1 - p) * (1 - t))
    return tp, fn, fp, tn

def crps(pred, target, pool_type='none', scale=1):
    normal = torch.distributions.Normal(0, 1)
    frac_sqrt_pi = 1 / np.sqrt(np.pi)
    eps = 1e-10
    if pred.ndim == 5:
        pred = pred.unsqueeze(1)
    b, n, t, c, h, w = pred.shape
    gt = einops.rearrange(target, 'b t c h w -> (b t) c h w')
    pr = einops.rearrange(pred,   'b n t c h w -> (b n t) c h w')
    if pool_type == 'avg':
        pr = F.avg_pool2d(pr, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif pool_type == 'max':
        pr = F.max_pool2d(pr, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    gt = einops.rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pr = einops.rearrange(pr, '(b n t) c h w -> b n t c h w', b=b, n=n)
    mean = torch.mean(pr, dim=1)
    std  = torch.std(pr,  dim=1) if n > 1 else torch.zeros_like(mean)
    normed = (mean - gt + eps) / (std + eps)
    cdf = normal.cdf(normed)
    pdf = normal.log_prob(normed).exp()
    val = (std + eps) * (normed * (2 * cdf - 1) + 2 * pdf - frac_sqrt_pi)
    return float(torch.mean(val).item())

def csi(pred, target, threshold, pool_type='none', scale=1):
    if pool_type in ('avg', 'max'):
        b, t = pred.shape[:2]
        fn = F.avg_pool2d if pool_type == 'avg' else F.max_pool2d
        p = einops.rearrange(pred,  'b t c h w -> (b t) c h w')
        g = einops.rearrange(target, 'b t c h w -> (b t) c h w')
        p = fn(p, scale, stride=scale)
        g = fn(g, scale, stride=scale)
        pred = einops.rearrange(p, '(b t) c h w -> b t c h w', b=b)
        target = einops.rearrange(g, '(b t) c h w -> b t c h w', b=b)
    tp, fn, fp, _ = _hit_miss_fa_cn(pred, target, threshold)
    return float((tp / (tp + fn + fp + _eps)).item())

def hss(pred, target, threshold, pool_type='none', scale=1):
    if pool_type in ('avg', 'max'):
        b, t = pred.shape[:2]
        fn = F.avg_pool2d if pool_type == 'avg' else F.max_pool2d
        p = einops.rearrange(pred,  'b t c h w -> (b t) c h w')
        g = einops.rearrange(target, 'b t c h w -> (b t) c h w')
        p = fn(p, scale, stride=scale)
        g = fn(g, scale, stride=scale)
        pred = einops.rearrange(p, '(b t) c h w -> b t c h w', b=b)
        target = einops.rearrange(g, '(b t) c h w -> b t c h w', b=b)
    tp, fn_, fp, tn = _hit_miss_fa_cn(pred, target, threshold)
    num = 2 * (tp * tn - fn_ * fp)
    den = (tp + fn_) * (fn_ + tn) + (tp + fp) * (fp + tn) + _eps
    return float((num / den).item())

def ssim(pred, target):
    cal = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    p = einops.rearrange(pred,  'b t c h w -> (b t) c h w')
    g = einops.rearrange(target, 'b t c h w -> (b t) c h w')
    return float(cal(p, g).item())

def psnr(pred, target):
    cal = PeakSignalNoiseRatio().to(pred.device)
    p = einops.rearrange(pred,  'b t c h w -> (b t) c h w')
    g = einops.rearrange(target, 'b t c h w -> (b t) c h w')
    total = 0.0
    for i in range(p.shape[0]):
        total += cal(p[i:i+1], g[i:i+1]).item()
    return float(total / p.shape[0])

def calc_metrics(pred, target):
    """
    pred and target shape == (b, t, c, h, w) in [0, 1] range.
    """
    pred = pred.detach()
    target = target.detach()
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    single = pred.mean(dim=1) if pred.ndim == 6 else pred
    results = {}
    
    # Base CRPS metrics
    results['CRPS'] = crps(pred, target, 'none', 1)
    results['CRPS_4'] = crps(pred, target, 'avg', 4)
    results['CRPS_16'] = crps(pred, target, 'avg', 16)
    
    # Base SSIM and PSNR
    results['SSIM'] = ssim(single, target)
    results['PSNR'] = psnr(single, target)
    
    # Base CSI and HSS for all thresholds and pools
    thresholds = [16/255, 74/255, 133/255, 160/255, 181/255, 219/255]
    for i, th in enumerate(thresholds):
        results[f'CSI_{i}'] = csi(single, target, th, 'none', 1)
        results[f'CSI_{i}_4'] = csi(single, target, th, 'avg', 4)
        results[f'CSI_{i}_16'] = csi(single, target, th, 'avg', 16)
        results[f'HSS_{i}'] = hss(single, target, th, 'none', 1)
        results[f'HSS_{i}_4'] = hss(single, target, th, 'avg', 4)
        results[f'HSS_{i}_16'] = hss(single, target, th, 'avg', 16)
    
    # Paper metrics - SSIM and PSNR (same for all pools)
    results['paper_SSIM'] = results['SSIM']
    results['paper_PSNR'] = results['PSNR']
    
    # Paper metrics - CRPS
    results['paper_CRPS'] = results['CRPS']
    
    # Paper metrics - aggregated CSI and HSS by pool
    for pool_name, suffix in [('POOL1', ''), ('POOL4', '_4'), ('POOL16', '_16')]:
        csi_vals = [results[f'CSI_{i}{suffix}'] for i in range(6)]
        hss_vals = [results[f'HSS_{i}{suffix}'] for i in range(6)]
        
        results[f'paper_CSI_M_{pool_name}'] = float(np.mean(csi_vals))
        results[f'paper_CSI_181_{pool_name}'] = results[f'CSI_4{suffix}']
        results[f'paper_CSI_219_{pool_name}'] = results[f'CSI_5{suffix}']
        results[f'paper_HSS_{pool_name}'] = float(np.mean(hss_vals))
    
    return results

if __name__ == '__main__':
    pred = torch.rand(2, 10, 1, 64, 64)
    target = torch.rand(2, 10, 1, 64, 64)
    metrics = calc_metrics(pred, target)
    
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')