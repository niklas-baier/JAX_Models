import numpy as np
from datasets import load_dataset
def load_mnist_hf(split: str, batch_size: int = 128, shuffle: bool = True):
    """
    Loads MNIST using Hugging Face datasets and returns a generator (DataLoader).
    """
    # Load the specific split
    ds = load_dataset("mnist", split=split)
    
    # Preprocess: Normalize and add channel dimension
    def transform(examples):
        # Convert list of images to 4D float32 array [Batch, H, W, 1]
        imgs = np.array(examples["image"], dtype=np.float32) / 255.0
        examples["image"] = imgs[..., None]
        return examples

    ds = ds.with_transform(transform)
    
    # Define the generator logic (The "DataLoader")
    def dataloader():
        # Shuffle indices if training
        indices = np.arange(len(ds))
        if shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = ds[batch_indices]
            # Convert to dictionary of arrays (standard JAX format)
            yield {
                "image": batch["image"],
                "label": np.array(batch["label"], dtype=np.int32).reshape(-1)
            }
            
    return dataloader
