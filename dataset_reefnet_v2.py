from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import torchvision

class ReefNetDataSet(Dataset):
    def __init__(self, csv_file, split='train', label_column='Experimental_label', path_column='patch_path',
                 split_column='split', transform=None, verbose=True):
        """
        Args:
            csv_file (str): Path to the CSV file.
            split (str): Which split to use ('train', 'val', or 'test'). Default: 'train'.
            label_column (str): Column name for labels. Default: 'genus'.
            path_column (str): Column name for the image paths. Default: 'patch_path'.
            split_column (str): Column name containing split info. Default: 'split'.
            transform: torchvision transforms or any callable for preprocessing the image.
            verbose (bool): If True, prints details about classes and sample counts.
        """
        self.csv_file = csv_file
        self.split = split
        self.label_column = label_column
        self.path_column = path_column
        self.split_column = split_column
        self.transform = transform
        self.verbose = verbose

        if self.verbose:
            print("Loading data from CSV file:", csv_file)
        full_data = pd.read_csv(csv_file)

        if self.verbose:
            # Show row count per split
            for sp in sorted(full_data[self.split_column].unique()):
                count = len(full_data[full_data[self.split_column] == sp])
                print(f"Number of rows in '{sp}' split: {count}")
            # Show class information
            unique_labels = sorted(full_data[label_column].unique())
            print("Number of unique classes:", len(unique_labels))
            print("Unique classes:", unique_labels)
            print("Class distribution:")
            print(full_data[label_column].value_counts())
        
        # Create a mapping from label to index (sorted alphabetically)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(full_data[label_column].unique()))}
        if self.verbose:
            print("Label to index mapping:", self.label_to_index)
        
        # Filter the CSV for the required split
        self.data = full_data[full_data[self.split_column] == split].copy()
        if self.verbose:
            print(f"Number of samples in the '{split}' split:", len(self.data))
        
        # Create a new column for the label index
        self.data['label_idx'] = self.data[label_column].map(self.label_to_index)
        self.nb_classes = len(self.label_to_index)
        self.class_counts = self.data['label_idx'].value_counts().sort_index().tolist()
        if self.verbose:
            print("Total number of classes:", self.nb_classes)
            print("Class counts for the current split:", self.class_counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path and label index from the CSV row
        img_path = self.data.iloc[idx][self.path_column]
        label = self.data.iloc[idx]['label_idx']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image at index {idx} from path {img_path}. Exception: {e}")
            raise e

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        # Check for invalid tensor values (if a tensor is returned)
        if isinstance(image, torch.Tensor):
            if torch.isnan(image).any() or torch.isinf(image).any():
                error_msg = f"[ERROR] Invalid image tensor at index {idx} (NaNs or Infs detected). Path: {img_path}"
                print(error_msg)
                # Optionally, you can inspect the bad tensor by saving it
                torchvision.utils.save_image(image, f"bad_input_{idx}.png")
                raise ValueError(error_msg)
        
        return image, int(label)

    def __repr__(self):
        return (f"ReefNetDataSet(csv_file='{self.csv_file}', split='{self.split}', samples={len(self)}, "
                f"classes={self.nb_classes})")