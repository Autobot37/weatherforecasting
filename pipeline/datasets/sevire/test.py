from pipeline.datasets.sevire.sevir import SEVIRLightningDataModule
data2 = SEVIRLightningDataModule(dataset_name="sevir_lr", num_workers=2, batch_size=8, seq_len=12, stride=20, layout='NTHW')
data2.setup()
data2.prepare_data()
train_loader2 = data2.train_dataloader()
from tqdm import tqdm
idx = 1000
for batch_idx, batch in enumerate(tqdm(train_loader2)):
    if batch_idx == 0:
        print(f"Batch {batch_idx} shape: {batch['vil'].shape}")
        print(f"Batch {batch_idx} keys: {batch.keys()}")
    if batch_idx >= idx:
        break
    