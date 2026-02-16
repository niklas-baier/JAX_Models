import os
import orbax.checkpoint as orbax
import shutil
import jax
import jax.numpy as jnp
import jax.profiler
import flax.nnx as nnx
import optax
import numpy as np
from tqdm import tqdm
from functools import partial
import einops

# PyTorch for data loading only
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Optional

# --- Import for metrics ---
from clu import metrics

PyTree = Any

# --- Model Definition (NNX) ---
class AttentionBlock(nnx.Module):
    """NNX version of the Attention Block."""
    def __init__(self, num_heads: int, embed_dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.qkv_projection = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim * 3,
            use_bias=False,
            rngs=rngs
        )
        self.output_projection = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            use_bias=False,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, *, train: bool, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        # Project to Q, K, V
        qkv = self.qkv_projection(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Rearrange for multi-head attention
        rearrange = partial(einops.rearrange, pattern='b s (h d) -> b h s d', h=self.num_heads)
        q, k, v = rearrange(q), rearrange(k), rearrange(v)

        # Scaled Dot-Product Attention
        attention_logits = jnp.einsum('b h s d, b h t d -> b h s t', q, k) / jnp.sqrt(q.shape[-1])
        # for sparse attention
        #sparse_mask = ...
        attention_logits = attention_logits #+ sparse_mask
        attention_weights = nnx.softmax(attention_logits, axis=-1)
        attention_output = jnp.einsum('b h s t, b h t d -> b h s d', attention_weights, v)

        # Combine heads and apply final projection
        attention_output = einops.rearrange(attention_output, 'b h s d -> b s (h d)')
        attention_output = self.output_projection(attention_output)

        # Only apply dropout during training with valid rngs
        if train and rngs is not None:
            attention_output = self.dropout(attention_output, deterministic=False, rngs=rngs)

        return attention_output
class SparseAttentionBlock(AttentionBlock):
    def __call__(self, x:jnp.ndarray, *, train:bool, rngs: Optional[nnx.Rngs]=None) -> jnp.ndarray:
        qkv = self.qkv_projection(x)
        q, k,v = jnp.split(qkv, 3, axis=-1)
        rearrange = partial(einops.rearrange, 'b h s d -> b s (h d)', h=self.num_heads)
        q, k , v = rearrange(q), rearrange(k), rearrange(v)
        attention_logits = jnp.einsum('b h s d, b h t d -> b h s t', q, k)/ jnp.sqrt(q.shape[-1])
        WINDOW_SIZE =10
        S = q.shape[2]
        i = jnp.arange(S)[:, None]
        j = jnp.arange(S)[None, :]
        distance = jnp.abs(i-j)
        local_mask = distance <= WINDOW_SIZE
        numerical_mask = jnp.where(local_mask,0, -1e9)
        attention_logits = attention_logits + numerical_mask[None, None, :,:]
        attention_weights = nnx.softmax(attention_logits, axis=-1)
        attention_output = jnp.einsum('b h s t, b h t d -> b h s d', attention_weights, v)
        attention_output = einops.rearrange(attention_output, 'b h s d -> b s (h d)')
        attention_output = self.output_projection(attention_output)
        if train and rngs is not None:
            attention_output = self.dropout(attention_output, deterministic=False, rngs=rngs)
        return attention_output


class FFNModern(nnx.Module):
    def __init__(self, embed_dim:int, ffn_hidden_factor:int, dropout_rate:float, *, rngs:nnx.Rngs):
        self.ffn_hidden_factor = ffn_hidden_factor
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        hidden_dim = embed_dim * self.ffn_hidden_factor * 2 // 3
        self.linear1 = nnx.Linear(in_features=embed_dim, out_features=hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features = embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=self.dropout_rate)
        self.dropout2 = nnx.Dropout(rate=self.dropout_rate)
    def __call__(self,x:jnp.ndarray, train:bool, rngs:nnx.Rngs):
        if train and rngs is not None:
            x = self.dropout1(x, deterministic=False, rngs=rngs)
        gate = nnx.gelu(self.linear1(x))
        x = self.linear1(x)
        x = gate * x
        if train and rngs is not None:
           x = self.dropout1(x, deterministic=False, rngs=rngs)
        gate = nnx.gelu(self.linear2(x))
        x = self.linear2(x)
        x = gate*x
        if train and rngs is not None:
                   x = self.dropout1(x, deterministic=False, rngs=rngs)
        return x
class FFNBlock(nnx.Module):
    def __init__(self, embed_dim: int, ffn_hidden_factor: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.ffn_hidden_factor = ffn_hidden_factor
        self.dropout_rate = dropout_rate

        hidden_dim = embed_dim * self.ffn_hidden_factor

        self.linear1 = nnx.Linear(in_features=embed_dim, out_features=hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features=embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=self.dropout_rate)
        self.dropout2 = nnx.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, *, train: bool, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        x = self.linear1(x)
        x = nnx.gelu(x)

        # Only apply dropout during training with valid rngs
        if train and rngs is not None:
            x = self.dropout1(x, deterministic=False, rngs=rngs)

        x = self.linear2(x)

        if train and rngs is not None:
            x = self.dropout2(x, deterministic=False, rngs=rngs)

        return x

class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, ffn_hidden_factor: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.attn = AttentionBlock(
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
            rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.ffn = FFNBlock(
            embed_dim=embed_dim,
            ffn_hidden_factor=ffn_hidden_factor,
            dropout_rate=dropout_rate,
            rngs=rngs
        )
        self.modernffn = FFNModern(embed_dim=embed_dim, ffn_hidden_factor=ffn_hidden_factor, dropout_rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        # Attention sub-layer with residual connection
        res = x
        x = self.norm1(x)
        x = self.attn(x, train=train, rngs=rngs)
        x = x + res

        # FFN sub-layer with residual connection
        res = x
        x = self.norm2(x)
        #x = self.ffn(x, train=train, rngs=rngs)
        x = self.modernffn(x, train=train, rngs=rngs)
        x = x + res

        return x

class VisionTransformerCNN(nnx.Module):
    def __init__(self, num_classes: int, num_heads: int, dropout_rate: float, ffn_hidden_factor: int, *, rngs: nnx.Rngs):
        cnn_features_1 = 32
        cnn_features_2 = 64
        transformer_embed_dim = cnn_features_1

        # --- CNN Feature Extractor ---
        self.conv1 = nnx.Conv(in_features=1, out_features=cnn_features_1, kernel_size=(3, 3), padding='SAME', rngs=rngs)

        # --- Transformer Encoder ---
        self.transformer_block = TransformerBlock(
            embed_dim=transformer_embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ffn_hidden_factor=ffn_hidden_factor,
            rngs=rngs
        )

        # --- Final Classification Head ---
        self.conv2 = nnx.Conv(in_features=cnn_features_1, out_features=cnn_features_2, kernel_size=(3, 3), padding='SAME', rngs=rngs)

        # Input (28) -> Pool1 (14) -> Pool2 (7). Final spatial size is 7x7.
        final_flat_size = 7 * 7 * cnn_features_2
        self.dense_out = nnx.Linear(in_features=final_flat_size, out_features=num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:

        # --- CNN Feature Extractor ---
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (b, 14, 14, 32)

        # --- Transformer Encoder ---
        b, h, w, c = x.shape
        x = jnp.reshape(x, (b, h * w, c))  # Flatten spatial dimensions

        x = self.transformer_block(x, train=train, rngs=rngs)
        x = jnp.reshape(x, (b, h, w, c))  # Reshape back

        # --- Final Classification Head ---
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (b, 7, 7, 64)

        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = self.dense_out(x)

        return x

# --- Trainer Class ---

@partial(jax.profiler.annotate_function, name="pytorch_to_jax")
def pytorch_to_jax(batch: Tuple) -> Dict[str, jnp.ndarray]:
    """Converts a PyTorch batch to a JAX-compatible dictionary."""
    images = np.array(batch[0].numpy()).transpose((0, 2, 3, 1))
    return {'image': jnp.array(images), 'label': jnp.array(batch[1].numpy())}


class Trainer:
    """A class to encapsulate training and evaluation state and logic."""

    def __init__(self, optimizer: nnx.ModelAndOptimizer):
        # Split the optimizer into static graph and dynamic state
        self.optimizer_graph, self.optimizer_state = nnx.split(optimizer)

    # --- Training Methods ---

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.profiler.annotate_function, name="train_step")
    def train_step(
        self,
        optimizer_state: nnx.State,
        batch: Dict,
        dropout_rng: jax.random.PRNGKey
    ) -> Tuple[nnx.State, jnp.ndarray]:
        """JIT-compiled training step."""

        # Reconstruct the optimizer
        optimizer = nnx.merge(self.optimizer_graph, optimizer_state)

        def loss_fn(model: VisionTransformerCNN):
            logits = model(
                batch['image'],
                train=True,
                rngs=nnx.Rngs(dropout=dropout_rng)
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']
            ).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
        optimizer.update(grads)

        # Return the new state
        _, new_optimizer_state = nnx.split(optimizer)
        return new_optimizer_state, loss

    # --- Evaluation Methods ---

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.profiler.annotate_function, name="eval_step")
    def eval_step(
        self,
        optimizer_state: nnx.State,
        batch: Dict
    ) -> Dict[str, jnp.ndarray]:
        """JIT-compiled eval step returning logits and labels for clu."""

        # Reconstruct the optimizer
        optimizer = nnx.merge(self.optimizer_graph, optimizer_state)

        # Get logits
        logits = optimizer.model(batch['image'], train=False)

        # Return dict for clu.metrics
        return {
            'logits': logits,
            'labels': batch['label']
        }

    @partial(jax.profiler.annotate_function, name="evaluate")
    def evaluate(self, test_loader: DataLoader) -> float:
        """Runs evaluation over the test set using clu.metrics."""

        # Accumulate accuracy metric directly
        accumulated_accuracy = None

        # Use the *current* optimizer state for the whole loop
        current_optimizer_state = self.optimizer_state

        for batch in test_loader:
            jax_batch = pytorch_to_jax(batch)

            # Get model output (logits and labels)
            metric_updates = self.eval_step(current_optimizer_state, jax_batch)

            # Create accuracy metric for this batch
            batch_accuracy = metrics.Accuracy.from_model_output(**metric_updates)

            if accumulated_accuracy is None:
                accumulated_accuracy = batch_accuracy
            else:
                accumulated_accuracy = accumulated_accuracy.merge(batch_accuracy)

        # Compute final accuracy
        return accumulated_accuracy.compute()

    def get_model(self) -> VisionTransformerCNN:
        """Reconstructs and returns the final model."""
        return nnx.merge(self.optimizer_graph, self.optimizer_state).model

# --- Data Handling ---

def prepare_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepares MNIST DataLoaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    return train_loader, test_loader


# --- Main Training & Evaluation Loop ---

def train_and_evaluate(config: Dict):
    """Main function to run the training and evaluation."""
    train_loader, test_loader = prepare_data(config['batch_size'])

    # Initialize RNG
    rng = jax.random.PRNGKey(config['seed'])
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # --- Model & Optimizer Setup ---
    model = VisionTransformerCNN(
        num_classes=config['num_classes'],
        num_heads=config['num_heads'],
        dropout_rate=config['dropout_rate'],
        ffn_hidden_factor=config['ffn_hidden_factor'],
        rngs=nnx.Rngs(params=init_rng)
    )
    # Dry run
    dummy_input = jnp.ones((1, 28, 28, 1))
    model(dummy_input, train=False)

    tx = optax.adamw(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer = nnx.ModelAndOptimizer(model, tx, wrt=nnx.Param)

    # --- Create the Trainer ---
    trainer = Trainer(optimizer)

    # --- Profiler Setup ---
    os.makedirs("./traces", exist_ok=True)
    trace_path = "./traces/jax_profile_nnx_v2"
    print("🚀 Starting training...")
    jax.profiler.start_trace(trace_path)

    step = 0
    for epoch in range(config['num_epochs']):
        # --- Training Loop ---
        running_loss = 0.0
        train_steps = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch") as progress_bar:
            for batch in progress_bar:
                dropout_rng, step_rng = jax.random.split(dropout_rng)
                jax_batch = pytorch_to_jax(batch)

                with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
                    # Call train_step directly with current state
                    new_state, loss = trainer.train_step(trainer.optimizer_state,jax_batch, step_rng)
                    # Update the trainer's internal state
                    trainer.optimizer_state = new_state

                running_loss += loss
                train_steps += 1
                step += 1
                progress_bar.set_postfix({'loss': f'{running_loss / train_steps:.4f}'})

        # --- Evaluation ---
        test_accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1} | Test Accuracy: {test_accuracy*100:.2f}%")

    jax.profiler.stop_trace()

    # --- Checkpointing ---
    print("✅ Training complete. Saving checkpoint...")
    checkpoint_path = '/home/niklas/Documents/functional_programming/checkpoints_nnx'

    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

    checkpointer = orbax.PyTreeCheckpointer()

    # Save the trainer's final state
    checkpoint_data = {
         'optimizer_state': trainer.optimizer_state
    }
    checkpointer.save(checkpoint_path, checkpoint_data)

    print(f"Checkpoint saved to {checkpoint_path}")

    # Return the final model from the trainer
    return trainer.get_model()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    config = {
        'num_epochs': 3,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'seed': 42,
        'num_heads': 8,
        'dropout_rate': 0.1,
        'ffn_hidden_factor': 4,
        'num_classes': 10,
    }
    final_model = train_and_evaluate(config)
