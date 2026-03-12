import numpy as np
from datasets import load_dataset
import os
import glob
import numpy as np
from PIL import Image
def load_eurosat_manual(data_dir: str, batch_size: int = 32, split_ratio: float = 0.8):
    """
    Manually loads EuroSAT from a folder structure and returns
    (train_dataloader, test_dataloader).
    """
    # 1. Get all class folders (AnnualCrop, Forest, etc.)
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    all_filepaths = []
    all_labels = []

    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        file_list = glob.glob(os.path.join(cls_path, "*.jpg"))
        all_filepaths.extend(file_list)
        all_labels.extend([class_to_idx[cls_name]] * len(file_list))

    all_filepaths = np.array(all_filepaths)
    all_labels = np.array(all_labels)

    # 2. Shuffle and Split
    indices = np.arange(len(all_filepaths))
    np.random.seed(42) # Consistent split
    np.random.shuffle(indices)
    split_ratio_test = 1 - ((1 - split_ratio)/2)
    split_ratio_super_test = 1 - ((1-split_ratio_test)/2)
    split_idx_train_valid = int(len(indices) * split_ratio)
    split_idx_valid_test = int(len(indices) * split_ratio_test)
    split_idx_test_super_test = int(len(indices) * split_ratio_super_test)

    train_indices = indices[:split_idx_train_valid]
    valid_indices = indices[split_idx_train_valid:split_idx_valid_test]
    test_indices = indices[split_idx_valid_test:split_idx_test_super_test]
    super_test_indices = indices[split_idx_test_super_test:]

    def create_loader(idx_list, shuffle=True):
        def loader():
            current_indices = idx_list.copy()
            if shuffle:
                np.random.shuffle(current_indices)

            for i in range(0, len(current_indices), batch_size):
                batch_idx = current_indices[i : i + batch_size]
                batch_files = all_filepaths[batch_idx]
                batch_labels = all_labels[batch_idx]

                # Load and preprocess images: [Batch, H, W, C]
                images = []
                for f in batch_files:
                    img = Image.open(f).convert("RGB")
                    img = np.array(img, dtype=np.float32) / 255.0
                    images.append(img)

                yield {
                    "image": np.stack(images),
                    "label": np.array(batch_labels, dtype=np.int32)
                }
        return loader

    return create_loader(train_indices, shuffle=True), create_loader(valid_indices, shuffle=False, ),create_loader(test_indices, shuffle=True), create_loader(super_test_indices, shuffle=False)


def load_mnist_hf(split: str, batch_size: int = 128, shuffle: bool = True):
    """
    Loads MNIST using Hugging Face datasets and returns a generator (DataLoader).
    """
    ds = load_dataset("mnist", split=split)

    def transform(examples):
        imgs = np.array(examples["image"], dtype=np.float32) / 255.0
        examples["image"] = imgs[..., None]
        return examples

    ds = ds.with_transform(transform)

    def dataloader():
        indices = np.arange(len(ds))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = ds[batch_indices]
            yield {
                "image": batch["image"],
                "label": np.array(batch["label"], dtype=np.int32).reshape(-1)
            }

    return dataloader
