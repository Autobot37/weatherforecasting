from pipeline.datasets.sevire.sevir import SEVIRLightningDataModule
data2 = SEVIRLightningDataModule(dataset_name="sevir_lr", num_workers=8, batch_size=8, seq_len=12, stride=20, layout='NTHW')
data2.setup()
data2.prepare_data()
train_loader2 = data2.train_dataloader()
sample2 = next(iter(train_loader2))
import torch
data2_dict = {}
print(len(train_loader2))
from tqdm import tqdm
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    if epoch == 0:
        for idx, sample in tqdm(enumerate(train_loader2)):
            data = sample['vil']
            data2_dict[data] = idx
    else:
        for sample in tqdm(train_loader2):
            data = sample['vil']
            idx = -1
            for i, d in enumerate(data2_dict.keys()):
                if torch.allclose(d, data, atol=1e-12):
                    idx = i
                    break
            print(f"Found data at index: {idx}")