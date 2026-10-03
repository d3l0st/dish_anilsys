import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.dataset import get_dataloaders


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return Config(**raw)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DishCalorieModel(nn.Module):
    def __init__(
        self,
        text_model_name,
        image_model_name,
        projection_dim,
        dropout,
        max_text_length,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = timm.create_model(
            image_model_name, pretrained=True, num_classes=0
        )
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, projection_dim
        )
        self.image_projection = nn.Linear(
            self.image_encoder.num_features, projection_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2 + 1, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.encode_text = partial(
            tokenizer,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        self._freeze(self.text_encoder)
        self._freeze(self.image_encoder)
        for parameter in self.image_encoder.layer3.parameters():
            parameter.requires_grad = True
        for parameter in self.image_encoder.layer4.parameters():
            parameter.requires_grad = True

    @staticmethod
    def _freeze(module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        self.image_encoder.eval()
        return self

    def encode_image(self, images):
        encoder = self.image_encoder
        with torch.no_grad():
            features = encoder.conv1(images)
            features = encoder.bn1(features)
            features = encoder.act1(features)
            features = encoder.maxpool(features)
            features = encoder.layer1(features)
            features = encoder.layer2(features)
        features = encoder.layer3(features)
        features = encoder.layer4(features)
        return encoder.forward_head(features, pre_logits=True)

    def forward(self, images, texts, mass):
        tokens = self.encode_text(list(texts))
        tokens = {key: value.to(images.device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = self.text_encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).type_as(hidden)
            text_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1e-9
            )
        image_embedding = self.encode_image(images)
        text_features = self.text_projection(text_embedding)
        image_features = self.image_projection(image_embedding)
        fused = torch.cat([text_features, image_features, mass.view(-1, 1)], dim=1)
        return self.classifier(fused).squeeze(-1)


def denormalize_calories(values, dataset):
    return values * dataset.calories_std + dataset.calories_mean


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    seen = 0
    for batch in loader:
        images = batch["image"].to(device)
        mass = batch["mass"].to(device)
        calories = batch["calories"].to(device)
        predictions = model(images, batch["text"], mass)
        loss = criterion(predictions, calories)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = calories.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
    return total_loss / seen


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    seen = 0
    mae = torchmetrics.MeanAbsoluteError().to(device)
    for batch in loader:
        images = batch["image"].to(device)
        mass = batch["mass"].to(device)
        calories = batch["calories"].to(device)
        predictions = model(images, batch["text"], mass)
        loss = criterion(predictions, calories)
        predictions_kcal = denormalize_calories(predictions, loader.dataset)
        calories_kcal = denormalize_calories(calories, loader.dataset)
        mae.update(predictions_kcal, calories_kcal)
        batch_size = calories.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
    return total_loss / seen, mae.compute().item()


def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
        data_dir=cfg.DATA_DIR,
    )
    model = build_model(cfg, device)
    optimizer = AdamW(
        [
            {"params": model.image_encoder.layer3.parameters(), "lr": cfg.IMAGE_LR},
            {"params": model.image_encoder.layer4.parameters(), "lr": cfg.IMAGE_LR},
            {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
            {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
            {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    criterion = nn.SmoothL1Loss()

    save_dir = resolve_save_dir(cfg)
    os.makedirs(save_dir, exist_ok=True)

    history = []
    best_mae = float("inf")
    # SmoothL1 считается в z-score; умножаем на std, чтобы логи/график были в шкале ккал.
    loss_scale = train_loader.dataset.calories_std
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        train_loss = train_loss * loss_scale
        val_loss = val_loss * loss_scale
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(row)
        print(
            f"Эпоха {epoch}/{cfg.EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}"
        )
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump({"best_val_mae": best_mae, "history": history}, file, indent=2)
    print(f"Лучший MAE на проверке: {best_mae:.4f}")
    print(f"Веса: {os.path.join(save_dir, 'best.pt')}")
    print(f"Метрики: {metrics_path}")
    plot_training_history(history, os.path.join(save_dir, "training.png"))
    return history


def plot_training_history(history, save_path):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, [row["train_loss"] for row in history], marker="o", label="Train loss"
    )
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Val loss")
    plt.plot(epochs, [row["val_mae"] for row in history], marker="o", label="Val MAE")
    plt.title("Обучение модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Ккал")
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"График: {save_path}")


def resolve_save_dir(cfg):
    save_dir = cfg.SAVE_DIR
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(PROJECT_ROOT, save_dir)
    return save_dir


def build_model(cfg, device):
    model = DishCalorieModel(
        text_model_name=cfg.TEXT_MODEL,
        image_model_name=cfg.IMAGE_MODEL,
        projection_dim=cfg.PROJECTION_DIM,
        dropout=cfg.DROPOUT,
        max_text_length=cfg.MAX_TEXT_LENGTH,
    )
    return model.to(device)


@torch.no_grad()
def evaluate_test(config_path):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = os.path.join(resolve_save_dir(cfg), "best.pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Нет файла весов: {weights}. Сначала запустите train.")

    _, val_loader = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
        data_dir=cfg.DATA_DIR,
    )
    model = build_model(cfg, device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mae = torchmetrics.MeanAbsoluteError().to(device)
    rows = []
    for batch in val_loader:
        images = batch["image"].to(device)
        mass = batch["mass"].to(device)
        calories = batch["calories"].to(device)
        predictions = model(images, batch["text"], mass)
        predictions_kcal = denormalize_calories(predictions, val_loader.dataset)
        calories_kcal = denormalize_calories(calories, val_loader.dataset)
        mae.update(predictions_kcal, calories_kcal)
        predictions_kcal = predictions_kcal.cpu()
        calories_kcal = calories_kcal.cpu()
        for index in range(calories.size(0)):
            real = float(calories_kcal[index])
            predicted = float(predictions_kcal[index])
            rows.append(
                {
                    "dish_id": batch["dish_id"][index],
                    "ingredients": batch["text"][index],
                    "total_calories": real,
                    "prediction": predicted,
                    "abs_error": abs(predicted - real),
                }
            )
    return mae.compute().item(), rows
