import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def replace_ingredients(dishes, ingredients):
    id_to_name = dict(
        zip(
            ingredients["id"].map(lambda ingredient_id: f"ingr_{ingredient_id:010d}"),
            ingredients["ingr"],
        )
    )
    dishes = dishes.copy()
    dishes["ingredients"] = dishes["ingredients"].map(
        lambda ids: ", ".join(id_to_name[token] for token in str(ids).split(";"))
    )
    return dishes


def load_dishes(data_dir):
    dishes = pd.read_csv(os.path.join(data_dir, "dish.csv"))
    ingredients = pd.read_csv(os.path.join(data_dir, "ingredients.csv"))
    return replace_ingredients(dishes, ingredients)


def clean_dishes(dishes):
    cleaned = dishes.copy()
    ingredients = cleaned["ingredients"].astype(str).str.strip().str.lower()
    keep = (
        (cleaned["total_calories"] > 0)
        & (ingredients != "plate only")
        & (cleaned["total_mass"] < 2000)
    )
    return cleaned.loc[keep].reset_index(drop=True)


def build_transforms(image_size, train):
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class DishDataset(Dataset):
    def __init__(
        self,
        frame,
        data_dir,
        transform,
        mass_mean,
        mass_std,
        calories_mean,
        calories_std,
    ):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.mass_mean = float(mass_mean)
        self.mass_std = float(mass_std)
        self.calories_mean = float(calories_mean)
        self.calories_std = float(calories_std)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        dish_id = row["dish_id"]
        img_path = os.path.join(self.data_dir, "images", str(dish_id), "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image=np.array(image))["image"]
        mass = (float(row["total_mass"]) - self.mass_mean) / self.mass_std
        calories = (
            float(row["total_calories"]) - self.calories_mean
        ) / self.calories_std
        return {
            "image": image,
            "text": row["ingredients"],
            "mass": np.float32(mass),
            "calories": np.float32(calories),
            "dish_id": dish_id,
        }


def get_dataloaders(batch_size=32, image_size=224, num_workers=0, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")

    dishes = clean_dishes(load_dishes(data_dir))
    train_frame = dishes[dishes["split"] == "train"]
    val_frame = dishes[dishes["split"] == "test"]
    mass_mean = float(train_frame["total_mass"].mean())
    mass_std = float(train_frame["total_mass"].std())
    calories_mean = float(train_frame["total_calories"].mean())
    calories_std = float(train_frame["total_calories"].std())

    train_dataset = DishDataset(
        train_frame,
        data_dir,
        build_transforms(image_size, train=True),
        mass_mean,
        mass_std,
        calories_mean,
        calories_std,
    )
    val_dataset = DishDataset(
        val_frame,
        data_dir,
        build_transforms(image_size, train=False),
        mass_mean,
        mass_std,
        calories_mean,
        calories_std,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
