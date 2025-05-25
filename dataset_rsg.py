from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import torchvision

class RSGPatchCSVLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        print("Loading data from CSV file:", csv_file)
        full_data = pd.read_csv(csv_file)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(full_data['genus'].unique()))}
        self.data = pd.read_csv('[Al-Wajh data csv file path]', index_col=0)
        self.rsg_genera_list =[
            "Porites",
            "Acropora",
            "Pocillopora",
            "Montipora",
            "Goniastrea",
            "Echinopora",
            "Stylophora",
            "Favites",
            "Lobophyllia",
            "Seriatopora",
            "Galaxea",
            "Astreopora",
            "Tubastraea",
            "Plerogyra"
        ]
        self.data = self.data[self.data['Our_Labels'].isin(self.rsg_genera_list)]
        print("Number of samples:", len(self.data))
        print("Number of classes:", len(self.label_to_index))
        print("Number of classes in RSG:", len(self.data['Our_Labels'].unique().tolist()))
        print("Classes:", self.label_to_index)
        self.data['label_idx'] = self.data['Our_Labels'].map(self.label_to_index)
        self.nb_classes = len(self.label_to_index)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"RSG (samples={len(self)}, classes={self.nb_classes})"

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['patch_path']
        label = self.data.iloc[idx]['label_idx']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # ✅ Check only after transform (i.e., once it's a Tensor)
        if isinstance(image, torch.Tensor):
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"[ERROR] Invalid image tensor at index {idx}: NaNs or Infs detected.")
                print(f"Path: {img_path}")
                torchvision.utils.save_image(image, f"bad_input_{idx}.png")
                raise ValueError("Invalid tensor found")

        return image, int(label)

