# -*- coding: utf-8 -*-
"""
train_bigru_transformer.py
IMU 감정 인식 – BiGRU + Transformer Encoder
(Adam, 추가 Dropout, EarlyStopping, EPOCHS=110)
"""
import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ------------------------------------------------- 설정
SEED = 42

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
set_seed()

device   = torch.device('mps' if torch.cuda.is_available() else 'cpu')
BASE_DIR = "./dataset_split_7_16"
SAVE_DIR = "./bigru_transformer_result"; os.makedirs(SAVE_DIR, exist_ok=True)
CHANNELS = ['ax1','ay1','az1','gx1','gy1','gz1','ax2','ay2','az2','gx2','gy2','gz2']
SEQ_LEN  = 150
BATCH    = 32
EPOCHS   = 100
PATIENCE = 10

# ------------------------------------------------- 증강 함수들 (강건한 조합)
def jitter(x, sigma=0.1):
    noise = np.random.normal(0, sigma, x.shape)
    return (x + noise).astype(np.float32)

def shift(x, max_shift=20):
    shift_amt = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift_amt, axis=0).astype(np.float32)

def scaling(x, sigma=0.2):
    factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
    return (x * factor).astype(np.float32)

def random_crop(x, crop_ratio=0.8):
    crop_len = int(SEQ_LEN * crop_ratio)
    start = np.random.randint(0, SEQ_LEN - crop_len + 1)
    cropped = x[start:start+crop_len]
    pad_len = SEQ_LEN - len(cropped)
    pad = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
    return np.vstack([cropped, pad])

AUGMENT_METHODS = [jitter, shift, scaling, random_crop]


def apply_augmentations(x, n_aug=1):
    methods = random.sample(AUGMENT_METHODS, k=n_aug)
    for fn in methods:
        x = fn(x)
    return x


def visualize_augmentation(example, methods, save_path):
    fig, axs = plt.subplots(len(methods)+1, 1, figsize=(10, 2*(len(methods)+1)))
    axs[0].plot(example)
    axs[0].set_title("Original")
    for i, method in enumerate(methods):
        aug = method(example.copy())
        axs[i+1].plot(aug)
        axs[i+1].set_title(f"{method.__name__}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------------------------------------- Dataset (복수 증강 적용)
class EmotionDataset(Dataset):
    def __init__(self, root, enc, augment=False, augment_prob=0.3):
        xs, ys = [], []
        self.augment = augment
        self.augment_prob = augment_prob
        for lbl in sorted(os.listdir(root)):
            p = os.path.join(root, lbl)
            if not os.path.isdir(p): continue
            for f in os.listdir(p):
                if not f.endswith('.csv'): continue
                df = pd.read_csv(os.path.join(p, f))
                if not set(CHANNELS).issubset(df.columns): continue
                arr = df[CHANNELS].values.astype(np.float32)
                arr = arr[:SEQ_LEN] if len(arr) >= SEQ_LEN else np.vstack([arr, np.zeros((SEQ_LEN-len(arr), len(CHANNELS)),np.float32)])
                xs.append(arr); ys.append(lbl)
        self.X = np.stack(xs); self.y = enc.transform(ys)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.augment and random.random() < self.augment_prob:
            x = apply_augmentations(x)
        return x, self.y[i]

# ------------------------------------------------- 모델
class BiGRU_Transformer(nn.Module):
    def __init__(self, in_ch, n_cls,
                 hid=128, gru_layers=4,
                 tf_layers=2, nhead=4, ff_dim=256,
                 drop_mid=0.2):
        super().__init__()
        self.bigru = nn.GRU(in_ch, hid, gru_layers,
                            batch_first=True, bidirectional=True)
        self.drop_mid = nn.Dropout(drop_mid)
        d_model = hid*2
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff_dim,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, tf_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model,128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,n_cls))
    def forward(self,x):
        g,_ = self.bigru(x)
        g   = self.drop_mid(g)
        t   = self.tr(g)
        f   = self.norm(t + g)
        return self.head(f.transpose(1,2))

# ------------------------------------------------- Helper
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, yt, yp = 0, [], []
    with torch.set_grad_enabled(train):
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            out   = model(xb); loss = crit(out,yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            yt.extend(yb.cpu().numpy()); yp.extend(out.argmax(1).cpu().numpy())
    return tot/len(loader), yt, yp

def plot_loss(tr,vl,path):
    plt.figure(); plt.plot(tr,label='train'); plt.plot(vl,label='val')
    plt.title('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

def save_cm(y_t,y_p,lbls,path):
    cm = confusion_matrix(y_t,y_p)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=lbls,yticklabels=lbls,cmap='Blues')
    plt.xlabel('Pred'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(path); plt.close()

def per_class_acc(y_true,y_pred,n_cls):
    y_true,y_pred = np.array(y_true), np.array(y_pred)
    return [(y_pred[y_true==c]==c).mean() if (y_true==c).any() else 0.0
            for c in range(n_cls)]


# ------------------------------------------------- Main

def main():
    enc = LabelEncoder(); enc.fit(sorted(os.listdir(f"{BASE_DIR}/train")))
    n_cls = len(enc.classes_)
    ld_tr = DataLoader(EmotionDataset(f"{BASE_DIR}/train", enc, augment=True, augment_prob=0.7), BATCH, True)
    ld_vl = DataLoader(EmotionDataset(f"{BASE_DIR}/val", enc), BATCH, False)
    ld_te = DataLoader(EmotionDataset(f"{BASE_DIR}/test", enc), BATCH, False)

    # -------------------------- 증강 시각화 예시 ----------------------------
    vis_save_path = os.path.join(SAVE_DIR, "augmentation_examples.png")
    sample_x, _ = next(iter(ld_tr))
    visualize_augmentation(sample_x[0].cpu().numpy(), AUGMENT_METHODS, vis_save_path)

if __name__ == "__main__":
    main()