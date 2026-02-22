from src.my_app.models import SimpleModel
from src.my_app.trainer import Trainer
from src.my_app.data import load_mnist_hf
from src.my_app.utils import setup_checkpointing, save_state
from src.my_app.logging import TensorBoardLogger
import tensorflow as tf
from flax import nnx
import yaml
import optax
import jax
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    cfg = load_config()
    m_cfg = cfg['model']  # contains hyperparams of the model
    t_cfg = cfg['training'] # contains hyperparam of the training
    train_loader_factory = load_mnist_hf('train', batch_size=t_cfg['batch_size'], shuffle=True)
    test_loader_factory = load_mnist_hf('test', batch_size=t_cfg['batch_size'], shuffle=False)
    logger = TensorBoardLogger(log_dir=t_cfg['log_dir'])
    model = SimpleModel(din=m_cfg['din'], dout=m_cfg['dout'], rngs=nnx.Rngs(m_cfg['rng_num']))

    test_loader = load_mnist_hf('test', batch_size=t_cfg['batch_size'], shuffle=False)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(t_cfg['learning_rate']))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average(argname='loss'), accuracy=nnx.metrics.Accuracy())
    trainer = Trainer(model, optimizer, metrics, logger=logger)#uncomment the logger
    cp_manager = setup_checkpointing(t_cfg['checkpoint_dir'])
    #jax.profiler.start_trace(t_cfg['trace_dir'])
    for epoch in range(t_cfg['epochs']):

        if trainer.logger:
            train_results = trainer.run_epoch(train_loader_factory(), is_training=True)
            trainer.logger.log_metrics(train_results, step=epoch, prefix='train')
            trainer.logger.log_layer_diagnostics(trainer.model, step=epoch)
            val_results = trainer.run_epoch(test_loader_factory(), is_training=False)

            trainer.logger.log_metrics(train_results, step=epoch, prefix='validation')
        print(f"Epoch {epoch+1}: Train Acc {train_results['accuracy']:.4f} | Val Acc {val_results['accuracy']:.4f}")

        if val_results['accuracy'] > trainer.best_acc:
            trainer.best_acc = val_results['accuracy']
            save_state(cp_manager, epoch, model, optimizer, float(trainer.best_acc))
            print("Waiting for final checkpoints to sync...")
            cp_manager.wait_until_finished()
            print("  ★ Checkpoint Saved")
    #jax.profiler.stop_trace()
