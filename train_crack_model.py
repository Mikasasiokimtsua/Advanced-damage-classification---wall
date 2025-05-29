import os
import pandas as pd

def parse_labels_from_folder(folder_name):

    idxs = [int(t.split('_')[0]) for t in folder_name.split() if '_' in t and t.split('_')[0].isdigit()]
    return idxs

def make_multilabel_csv(data_dir, csv_path):
    records = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path): continue
        label_indices = parse_labels_from_folder(folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            full_path = os.path.join(folder_path, fname)
            row = {'filepath': full_path}
            for i in range(11):
                row[f'label_{i}'] = 1 if i in label_indices else 0
            records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}, shape:", df.shape)

make_multilabel_csv("datasets/crack/train", "train_labels.csv")
make_multilabel_csv("datasets/crack/valid", "valid_labels.csv")

import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 11
BATCH_SIZE, EPOCHS, LR = 32, 500, 1e-4 # 20 1e-4

train_csv = "train_labels.csv"
valid_csv = "valid_labels.csv"

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

class CrackMultiLabelDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert('RGB')
        if self.transform: img = self.transform(img)
        labels = torch.tensor(row[[f'label_{i}' for i in range(num_labels)]].values.astype('float32'))
        return img, labels

train_set = CrackMultiLabelDataset(train_csv, train_tfm)
val_set   = CrackMultiLabelDataset(valid_csv, val_tfm)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_labels)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = 1e9
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"[{epoch+1:02d}] train loss={avg_loss:.4f}, val loss={avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_crack_model.pth")
        print("Save best_crack_model.pth")
