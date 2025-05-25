from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import torchvision

class CoralPatchCSVLoader(Dataset):
    def __init__(self, csv_file, is_train=True, is_test=False, transform=None):
        print("Loading data from CSV file:", csv_file)
        full_data = pd.read_csv(csv_file)
        print("Data loaded. Preprocessing...")
        # print the number of rows in each split 
        print("Number of rows in train split:", len(full_data[full_data['split'] == 'train']))
        print("Number of rows in test split:", len(full_data[full_data['split'] == 'test']))
        print("Number of rows in val split:", len(full_data[full_data['split'] == 'val']))
        # print the number of unique classes
        print("Number of unique classes:", len(full_data['genus'].unique()))
        # print the unique classes
        print("Unique classes:", full_data['genus'].unique())
        # print the number of samples in each class
        print("Number of samples in each class:")
        print(full_data['genus'].value_counts())
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(full_data['genus'].unique()))}
        if is_train:
            self.data = full_data[full_data['split'] == 'train'].copy()
        elif is_test:
            self.data = full_data[full_data['split'] == 'test'].copy()
        else:
            self.data = full_data[full_data['split'] == 'val'].copy()

        print("Number of samples in the current split:", len(self.data))
        print("Number of classes in label_to_index:", len(self.label_to_index))
        print("Classes from label_to_index dict:", self.label_to_index)
        self.data['label_idx'] = self.data['genus'].map(self.label_to_index)
        self.nb_classes = len(self.label_to_index)
        self.transform = transform
        self.class_counts = self.data['label_idx'].value_counts().sort_index().tolist()
        print("Class counts for current split:", self.class_counts)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"CoralPatchCSVLoader(samples={len(self)}, classes={self.nb_classes})"

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

