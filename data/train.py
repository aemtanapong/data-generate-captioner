import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

from torchvision import transforms, models
SEED = 50

# random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)
# =========================
# CONFIG
# =========================

CSV_PATH = "train_data.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4

NUM_DISTRICTS = 50
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(43)
print(DEVICE)
# =========================
# DATASET
# =========================

class RainDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)

        self.transform = transform

        # column 1 = frame image
        self.image_paths = self.df.iloc[:, 1]

        # remove first 2 columns
        self.labels_df = self.df.iloc[:, 2:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.image_paths.iloc[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            self.labels_df.iloc[idx].values.astype(np.int64)
        )

        return image, labels

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# FULL DATASET
# =========================

full_dataset = RainDataset(
    CSV_PATH,
    transform=transform
)

# =========================
# SPLIT TRAIN / VALID
# =========================

train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# =========================
# DATALOADER
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =========================
# MODEL
# =========================

class RainResNet50(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet50(
            weights="IMAGENET1K_V1"
        )
        for idx, param in enumerate(self.backbone.parameters()):
            if idx <  -1:
                param.requires_grad = False
            # print(idx)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            NUM_DISTRICTS * NUM_CLASSES
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(
            -1,
            NUM_DISTRICTS,
            NUM_CLASSES
        )

        return x
class RainVGG16(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.vgg16(
            weights="IMAGENET1K_V1"
        )

        in_features = self.backbone.classifier[6].in_features

        self.backbone.classifier[6] = nn.Linear(
            in_features,
            NUM_DISTRICTS * NUM_CLASSES
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(
            -1,
            NUM_DISTRICTS,
            NUM_CLASSES
        )

        return x
class RainConvNext(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.convnext_tiny(
            weights="IMAGENET1K_V1"
        )

        in_features = self.backbone.classifier[2].in_features

        self.backbone.classifier[2] = nn.Linear(
            in_features,
            NUM_DISTRICTS * NUM_CLASSES
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(
            -1,
            NUM_DISTRICTS,
            NUM_CLASSES
        )

        return x
class RainMobileNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.mobilenet_v3_large(
            weights="IMAGENET1K_V1"
        )

        in_features = self.backbone.classifier[3].in_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        # replace classifier
        self.backbone.classifier[3] = nn.Sequential(

            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(
                512,
                NUM_DISTRICTS * NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(
            -1,
            NUM_DISTRICTS,
            NUM_CLASSES
        )

        return x
class RainEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b7(
            weights="IMAGENET1K_V1"
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier[1] = nn.Linear(
            in_features,
            NUM_DISTRICTS * NUM_CLASSES
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(
            -1,
            NUM_DISTRICTS,
            NUM_CLASSES
        )

        return x
rain_model = RainResNet50
model = rain_model().to(DEVICE)

# =========================
# LOSS / OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LR
)

best_val_loss = 999999

# =========================
# TRAIN
# =========================

for epoch in range(EPOCHS):

    # =====================
    # TRAIN
    # =====================

    model.train()

    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = 0

        for i in range(NUM_DISTRICTS):

            loss += criterion(
                outputs[:, i, :],
                labels[:, i]
            )

        loss = loss / NUM_DISTRICTS

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    val_loss = 0

    total_correct = 0
    total_count = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = 0

            for i in range(NUM_DISTRICTS):

                loss += criterion(
                    outputs[:, i, :],
                    labels[:, i]
                )

            loss = loss / NUM_DISTRICTS

            val_loss += loss.item()

            # =====================
            # ACCURACY
            # =====================

            preds = outputs.argmax(dim=-1)

            correct = (
                preds == labels
            ).sum().item()

            total_correct += correct
            total_count += labels.numel()

    val_loss /= len(val_loader)

    val_acc = total_correct / total_count

    # =====================
    # SAVE BEST MODEL
    # =====================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "nbest_rain_model.pth"
        )

        print("Best model saved")

    # =====================
    # PRINT
    # =====================

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_acc:.4f}"
    )

    print("-" * 50)