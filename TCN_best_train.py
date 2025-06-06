# Improved TCN training script with better regularization, task weighting, GELU activation, and scheduler
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== CONFIG ======
INFO_CSV = "./39_Training_Dataset/train_info.csv"
DATA_FOLDER = "./39_Training_Dataset/train_data"
MODEL_SAVE_PATH = "best_tcn_model.pth"
MAX_SEQ_LEN = 500
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Dataset ======
class TCNDataset(Dataset):
    def __init__(self, info_df, data_folder, max_len=500):
        self.samples = []
        self.labels = []
        for _, row in info_df.iterrows():
            uid = row['unique_id']
            path = os.path.join(data_folder, f"{uid}.txt")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=r'\s+', header=None,
                             names=["Ax","Ay","Az","Gx","Gy","Gz"], engine='python')
            data = df.values[:max_len]
            pad_len = max_len - len(data)
            if pad_len > 0:
                data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')
            # Normalization
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            self.samples.append(data.astype(np.float32))
            self.labels.append([
                1 if row['gender'] == 1 else 0,
                1 if row['hold racket handed'] == 1 else 0,
                int(row['play years']),
                {2: 0, 3: 1, 4: 2, 5: 3}[row['level']]
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

# ====== TCN Block ======
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.3)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.pad]
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.pad]
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        res = self.downsample(x)
        if res.shape[-1] != out.shape[-1]:
            res = res[:, :, -out.shape[-1]:]
        return out + res

# ====== TCN Model ======
class MultiTaskTCN(nn.Module):
    def __init__(self, input_dim=6, num_channels=[64, 128, 256, 256], kernel_size=5):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gender_head = nn.Linear(in_ch, 2)
        self.hold_head = nn.Linear(in_ch, 2)
        self.year_head = nn.Linear(in_ch, 3)
        self.level_head = nn.Linear(in_ch, 4)

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.tcn(x)
        pooled = self.pool(features).squeeze(-1)
        return {
            'gender': self.gender_head(pooled),
            'hold': self.hold_head(pooled),
            'year': self.year_head(pooled),
            'level': self.level_head(pooled)
        }

# ====== Training ======
def compute_auc(preds, labels, num_classes):
    probs = torch.softmax(preds, dim=1).detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    if num_classes == 2:
        return roc_auc_score(labels_np, probs[:, 1])
    else:
        return roc_auc_score(labels_np, probs, multi_class='ovr', average='macro')

def train_model(dataset):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_levels = [lbl[3] for lbl in dataset.labels]
    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, y_levels)):
        print(f"\n===== Fold {fold+1} =====")
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = MultiTaskTCN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        weights = {'gender': 1.0, 'hold': 1.0, 'year': 1.0, 'level': 2.0}

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = sum([weights[k] * criterion(out[k], y[:, i]) for i, k in enumerate(out.keys())])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds_all = {k: [] for k in ['gender','hold','year','level']}
        labels_all = {k: [] for k in ['gender','hold','year','level']}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                for i, k in enumerate(out.keys()):
                    preds_all[k].append(out[k])
                    labels_all[k].append(y[:, i])
        score_level = 0
        for k in preds_all:
            pred_cat = torch.cat(preds_all[k])
            label_cat = torch.cat(labels_all[k])
            acc = (pred_cat.argmax(dim=1) == label_cat).float().mean().item()
            auc = compute_auc(pred_cat, label_cat, pred_cat.shape[1])
            print(f"{k:>6} - Acc: {acc:.4f}, AUC: {auc:.4f}")
            if k == 'level':
                score_level = auc

        scheduler.step(score_level)

        if score_level > best_score:
            best_score = score_level
            best_model = model.state_dict()

    torch.save(best_model, MODEL_SAVE_PATH)
    print(f"\n最佳模型已儲存至: {MODEL_SAVE_PATH}，Level AUC: {best_score:.4f}")

# ====== Main ======
if __name__ == '__main__':
    info_df = pd.read_csv(INFO_CSV)
    dataset = TCNDataset(info_df, DATA_FOLDER, max_len=MAX_SEQ_LEN)
    train_model(dataset)
