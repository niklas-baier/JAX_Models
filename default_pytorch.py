import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.profiler
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
from functools import partial
import einops
from typing import Dict, Tuple, Any

# --- Model Definition (PyTorch) ---

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads

        # In PyTorch, nn.Linear requires in_features at init time
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_features = x.shape[-1]

        # Project to Q, K, V
        qkv = self.qkv_projection(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Rearrange for multi-head attention
        rearrange = partial(einops.rearrange, pattern='b s (h d) -> b h s d', h=self.num_heads)
        q, k, v = rearrange(q), rearrange(k), rearrange(v)

        # Scaled Dot-Product Attention
        attention_logits = torch.einsum('b h s d, b h t d -> b h s t', q, k) / torch.sqrt(torch.tensor(q.shape[-1]).float())
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_output = torch.einsum('b h s t, b h t d -> b h s d', attention_weights, v)

        # Combine heads and apply final projection
        attention_output = einops.rearrange(attention_output, 'b h s d -> b s (h d)')
        attention_output = self.output_projection(attention_output)

        # Dropout uses self.training (set by model.train() or model.eval())
        attention_output = self.dropout(attention_output)

        return attention_output

class FFNBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_hidden_factor: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * ffn_hidden_factor

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, ffn_hidden_factor: int):
        super().__init__()

        # Pre-norm architecture, matching the JAX code
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFNBlock(embed_dim=embed_dim, ffn_hidden_factor=ffn_hidden_factor, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + res

        # FFN sub-layer with residual connection
        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + res

        return x

class VisionTransformerCNN(nn.Module):
    def __init__(self, num_classes: int = 10, num_heads: int = 8, dropout_rate: float = 0.1, ffn_hidden_factor: int = 4):
        super().__init__()

        # --- CNN Feature Extractor ---
        # PyTorch nn.Conv2d requires (N, C, H, W)
        # We use padding='same' to match JAX's 'SAME' padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # -> (N, 32, 14, 14)

        # --- Transformer Encoder ---
        # The JAX TransformerBlock is applied to (N, H*W, C)
        # The input channel dim (32) becomes the embedding dim
        self.transformer_block = TransformerBlock(
            embed_dim=32,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ffn_hidden_factor=ffn_hidden_factor
        )

        # --- Final Classification Head ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # -> (N, 64, 7, 7)

        # Flattened size is 64 * 7 * 7
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (N, H, W, C) = (N, 28, 28, 1) to match JAX

        # --- CNN Feature Extractor ---
        # Permute to (N, C, H, W) for PyTorch Conv2d
        x = x.permute(0, 3, 1, 2) # -> (N, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x) # -> (N, 32, 14, 14)

        # --- Transformer Encoder ---
        # Permute back to (N, H, W, C) for Transformer
        x = x.permute(0, 2, 3, 1) # -> (N, 14, 14, 32)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c) # Flatten spatial dims -> (N, 196, 32)

        x = self.transformer_block(x) # -> (N, 196, 32)

        x = x.reshape(b, h, w, c) # Reshape back -> (N, 14, 14, 32)

        # --- Final Classification Head ---
        # Permute to (N, C, H, W) for final Conv2d
        x = x.permute(0, 3, 1, 2) # -> (N, 32, 14, 14)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x) # -> (N, 64, 7, 7)

        x = torch.flatten(x, 1) # Flatten -> (N, 64 * 7 * 7)
        x = self.fc(x) # -> (N, 10)

        return x

# --- Data Handling ---

def prepare_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepares MNIST DataLoaders. (Unchanged from JAX example)"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    return train_loader, test_loader

def batch_to_torch(batch: Tuple, device: torch.device) -> Dict[str, torch.Tensor]:
    """Converts a PyTorch (N, C, H, W) batch to (N, H, W, C) and moves to device."""
    images, labels = batch

    # Transpose images from (N, C, H, W) to (N, H, W, C) to match JAX model input
    images = images.permute(0, 2, 3, 1).to(device)
    labels = labels.to(device)
    return {'image': images, 'label': labels}


# --- Main Training & Evaluation Loop ---

def train_and_evaluate_torch(config: Dict):
    """Main function to run the training and evaluation."""

    # Set seed for reproducibility
    torch.manual_seed(config['seed'])

    # Setup device
    # This check is crucial. If torch.cuda.is_available() is False,
    # it means you have a CPU-only PyTorch install.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = prepare_data(config['batch_size'])

    # Model, Optimizer, Loss
    model = VisionTransformerCNN(
        num_classes=10,
        num_heads=8,
        dropout_rate=0.1,
        ffn_hidden_factor=4
    ).to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    print("🚀 Starting PyTorch training...")

    # Start profiler
    # --- MODIFIED FOR LEGACY PYTORCH ---
    # The old API uses boolean flags, not 'activities' or 'on_trace_ready'
    with torch.profiler.profile(
        record_shapes=True,
        use_cuda=use_cuda # Use the result from torch.cuda.is_available()
    ) as prof:
    # --- END MODIFICATION ---

        for epoch in range(config['num_epochs']):

            # --- Training Loop ---
            model.train() # Set model to training mode (enables dropout)
            running_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch") as progress_bar:
                for step, batch in enumerate(progress_bar):

                    torch_batch = batch_to_torch(batch, device)
                    images = torch_batch['image']
                    labels = torch_batch['label']

                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{running_loss / (step + 1):.4f}'})

                    # Signal profiler step
                    prof.step()

            # --- Evaluation Loop ---
            model.eval() # Set model to evaluation mode (disables dropout)
            accuracies = []
            with torch.no_grad(): # Disable gradient calculation
                for batch in test_loader:
                    torch_batch = batch_to_torch(batch, device)
                    images = torch_batch['image']
                    labels = torch_batch['label']

                    logits = model(images)
                    preds = torch.argmax(logits, dim=-1)
                    acc = (preds == labels).float().mean()
                    accuracies.append(acc.item())

            test_accuracy = np.mean(accuracies)
            print(f"Epoch {epoch+1} | Test Accuracy: {test_accuracy*100:.2f}%")

    # --- MODIFICATION FOR LEGACY PYTORCH ---
    # Manually export the trace *after* the profiling block has finished
    print(f"Writing profiler trace to /tmp/torch_profile_base.json")
    # This file can be loaded into chrome://tracing
    prof.export_chrome_trace("/tmp/torch_profile_base.json")
    # --- END MODIFICATION ---

    return model

if __name__ == "__main__":
    config = {
        'num_epochs': 3,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'seed': 42
    }
    final_model = train_and_evaluate_torch(config)
