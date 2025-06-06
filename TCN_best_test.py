# test_tcn_multitask.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from TCN_train import MultiTaskTCN

# ====== CONFIG ======
TEST_CSV = "./39_Test_Dataset/test_info.csv"
TEST_FOLDER = "./39_Test_Dataset/test_data"
MODEL_PATH = "best_tcn_model.pth"
OUTPUT_FILE = "test_predictions.csv"
MAX_SEQ_LEN = 500
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Dataset ======
class TCNTestDataset(Dataset):
    def __init__(self, info_df, data_folder, max_len=500):
        self.samples = []
        self.uids = []
        for _, row in info_df.iterrows():
            uid = row['unique_id']
            path = os.path.join(data_folder, f"{uid}.txt")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=r'\s+', header=None, names=["Ax","Ay","Az","Gx","Gy","Gz"], engine='python')
            data = df.values[:max_len]
            pad_len = max_len - len(data)
            if pad_len > 0:
                data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')
            # === 加入 Z-score 標準化 ===
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            self.samples.append(data.astype(np.float32))
            self.uids.append(uid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.uids[idx])

# ====== Inference ======
def predict():
    info_df = pd.read_csv(TEST_CSV)
    test_dataset = TCNTestDataset(info_df, TEST_FOLDER, max_len=MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MultiTaskTCN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    results = []
    with torch.no_grad():
        for x, uids in tqdm(test_loader, desc="預測中"):
            x = x.to(DEVICE)
            out = model(x)

            prob_g = torch.softmax(out['gender'], dim=1).cpu().numpy()
            prob_h = torch.softmax(out['hold'], dim=1).cpu().numpy()
            prob_y = torch.softmax(out['year'], dim=1).cpu().numpy()
            prob_l = torch.softmax(out['level'], dim=1).cpu().numpy()

            for i, uid in enumerate(uids):
                results.append({
                    "unique_id": int(uid.item()),
                    "gender": prob_g[i,1],
                    "hold racket handed": prob_h[i,1],
                    "play years_0": prob_y[i,0],
                    "play years_1": prob_y[i,1],
                    "play years_2": prob_y[i,2],
                    "level_2": prob_l[i,0],
                    "level_3": prob_l[i,1],
                    "level_4": prob_l[i,2],
                    "level_5": prob_l[i,3],
                })

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(OUTPUT_FILE, index=False, float_format='%.8f')
    print(f"預測完成，結果已保存至 {OUTPUT_FILE}")

# ====== Main ======
if __name__ == '__main__':
    predict()
