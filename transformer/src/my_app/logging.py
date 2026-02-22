# this is for visualization and logging only.
import tensorflow as tf
from flax import nnx
import jax
import jax.numpy as jnp
class TensorBoardLogger:
    def __init__(self,log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_metrics(self, metrics_dict, step, prefix=""):
            with self.writer.as_default():
                for name, value in metrics_dict.items():
                    tag = f"{prefix}/{name}" if prefix else name
                    tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
    def log_layer_diagnostics(self, model: nnx.Module, step: int):
        state = nnx.state(model)

        # Flatten the state into (leaves, treedef) and get paths
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)

        with self.writer.as_default():
            for path, val in leaves_with_paths:
                # Convert the jax key path to a readable string
                name = "/".join(
                    k.key if hasattr(k, 'key') else str(k)
                    for k in path
                )
                has_nan = jnp.any(jnp.isnan(val))
                tf.summary.scalar(f"stats/{name}/has_nan", int(has_nan), step=step)
                if not has_nan:
                    tf.summary.histogram(f"dist/{name}", val, step=step)
                    tf.summary.scalar(f"stats/{name}/mean", jnp.mean(val).item(), step=step)
                    tf.summary.scalar(f"stats/{name}/std", jnp.std(val).item(), step=step)
                    tf.summary.scalar(f"stats/{name}/max", jnp.max(val).item(), step=step)
                    tf.summary.scalar(f"stats/{name}/min", jnp.min(val).item(), step=step)
        self.writer.flush()
