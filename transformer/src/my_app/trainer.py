import jax
import optax
from flax import nnx
from functools import partial
from tqdm import tqdm
class Trainer(nnx.Module):
    def __init__(self, model, optimizer, metrics, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.best_acc = nnx.Variable(jax.numpy.array(0.0))
        self.step =  nnx.Variable(jax.numpy.array(0.0))
        self.logger = logger

    @nnx.jit
    @partial(jax.profiler.annotate_function, name='train_step')
    def train_step(self, batch):
        def loss_fn(model):
            logits = model(batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels = batch['label']).mean()
            return loss, logits

        (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)
        self.optimizer.update(self.model,grads)
        self.metrics.update(loss=loss, logits=logits, labels=batch['label'])

        new_step = self.step +1
        self.step = new_step

    @nnx.jit
    @partial(jax.profiler.annotate_function, name='val_step')
    def val_step(self, batch):
        logits = self.model(batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        self.metrics.update(loss=loss, logits=logits, labels=batch['label'])

    def run_epoch(self, loader, is_training=True):
        self.metrics.reset()

        desc = "Training" if is_training else "Validating"
        pbar = tqdm(loader, desc=desc, leave=False)

        for batch in pbar:
            if is_training:
                self.train_step(batch)
            else:
                self.val_step(batch)
            current_metrics = self.metrics.compute()
            pbar.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

        return self.metrics.compute()
