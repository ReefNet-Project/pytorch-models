# ReefNet Training Code

This repository contains data loading utilities used in the ReefNet project for training coral reef classification models using [timm](https://github.com/huggingface/pytorch-image-models), a PyTorch-based image modeling library. The provided dataset classes enable flexible, split-aware loading of coral patch images annotated with genus-level labels.

➡️ For model details, benchmarks, and data, visit the project page: [https://reefnet-project.github.io/reefnet-2025/](https://reefnet-project.github.io/reefnet-2025/)

---

## 📁 Directory Structure

```bash
pytroch-models/
│   ├── reefnet_dataset.py          # General-purpose loader
│   ├── dataset_reefnet_v2.py       # main data loader for classification tasks used in ReefNet code 
│   └── dataset_rsg.py         # Al-wajh dataset loader
│   ├── ...                         # rest of the files like the original repo
├── README.md
└── ... (integration with timm)
```

---

## 🔍 Overview of Data Loaders

### `ReefNetDataset`
- General-purpose dataset class supporting arbitrary label, path, and split column names.
- Outputs: `(image_tensor, label_index)`
- CSV must contain columns like `patch_path`, `split`, and your chosen label column (e.g., `genus` or `Experimental_label`).

### `CoralPatchCSVLoader`
- Simplified loader for standard `train/val/test` splits using `genus` as the label column.
- Prints summary statistics on load: class counts, unique labels, etc.

### `RSGPatchCSVLoader`
- Specialized loader for Al-wajh testing dataset.
- Filters samples based on a predefined list of genera (e.g., Porites, Acropora, Pocillopora).
- Use this for domain-specific generalization or Red Sea evaluation tasks.

---

## 🛠️ Usage

### Example (Standalone PyTorch Training)
```python
from loaders.reefnet_dataset import ReefNetDataSet
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ReefNetDataSet(
    csv_file='reefnet_metadata.csv',
    split='train',
    label_column='genus',
    path_column='patch_path',
    transform=transform
)

image, label = dataset[0]
print(image.shape, label)
```

---

## 📝 CSV Format
All loaders expect a CSV with at least these columns:
- `patch_path`: path to the coral patch image
- `genus` or `Experimental_label`: genus-level coral label
- `split`: one of `train`, `val`, or `test`

For `RSGPatchCSVLoader`, an additional column `Our_Labels` is used to filter classes.

---

## 📘 Citing ReefNet
If you use these loaders or any ReefNet data or models in your research, please cite:

```bibtex
@article{battach2025reefnet,
  title={ReefNet: A Large-scale, Taxonomically Enriched Dataset and Benchmark for Coral Reef Classification},
  author={Battach, Yahia and Felemban, Abdulwahab and Khan, Faizan Farooq and Radwan, Yousef A. and Li, Xiang and Silva, Luis and Suka, Rhonda and Gonzalez, Karla and Marchese, Fabio and Williams, Ivor D. and Jones, Burton H. and Beery, Sara and Benzoni, Francesca and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Maintainer
- [Yahia Battach](https://github.com/shakesBeardZ/) — Creator and Maintainer


---

## 📄 License
This code is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

## 🙋‍♀️ Questions?
Feel free to open an issue or visit our [project homepage](https://reefnet-project.github.io/reefnet-2025/) for more resources.
