from src.my_app.models import SimpleModel, AutoEncoder
from src.my_app.trainer import Trainer
from src.my_app.data import load_mnist_hf, load_eurosat_manual
from src.my_app.utils import setup_checkpointing, save_state, load_config
from src.my_app.logging import TensorBoardLogger
import tensorflow as tf
from flax import nnx
import yaml
import optax
import jax
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    cfg = load_config()
    m_cfg = cfg['model']  # contains hyperparams of the model
    t_cfg = cfg['training'] # contains hyperparam of the training
    d_cfg = cfg['dataset']
    data_path = d_cfg['path']

    train_loader_factory, valid_loader_factory, test_loader_factory, super_test_loader_factory = load_eurosat_manual(
        data_path,
        batch_size=t_cfg['batch_size']
    )
    logger = TensorBoardLogger(log_dir=t_cfg['log_dir'])
    #model = SimpleModel(din=m_cfg['din'], dout=m_cfg['dout'],image_shape=d_cfg['image_size'], rngs=nnx.Rngs(m_cfg['rng_num']))
    model = AutoEncoder(input_dim=1,num_classes=d_cfg['num_classes'],num_layers=4, num_channels=3, rngs=nnx.Rngs(m_cfg['rng_num']))

    test_loader = load_mnist_hf('test', batch_size=t_cfg['batch_size'], shuffle=False)
    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(t_cfg['learning_rate'])
        ),
        every_k_schedule=t_cfg['grad_accum_steps']
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(argname='total_loss'), # Total combined loss
        rec_loss=nnx.metrics.Average(argname='rec_loss'), # Just the L2 part
        ce_loss = nnx.metrics.Average(argname='ce_loss'),
        accuracy=nnx.metrics.Accuracy()
    )
    trainer = Trainer(model, optimizer, metrics, logger=logger)#uncomment the logger
    cp_manager = setup_checkpointing(t_cfg['checkpoint_dir'])
    #jax.profiler.start_trace(t_cfg['trace_dir'])
    trainer.logger.log_number_of_trainable_parameters(trainer.model)
    for epoch in range(t_cfg['epochs']):

        if trainer.logger:
            train_results = trainer.run_epoch(train_loader_factory(), is_training=True)
            trainer.logger.log_metrics(train_results, step=epoch, prefix='train')
            trainer.logger.log_layer_diagnostics(trainer.model, step=epoch)
            val_results = trainer.run_epoch(test_loader_factory(), is_training=False)

            trainer.logger.log_metrics(val_results, step=epoch, prefix='validation')
        print(f"Epoch {epoch+1}: Train Acc {train_results['accuracy']:.4f} | Val Acc {val_results['accuracy']:.4f}")

        if val_results['accuracy'] > trainer.best_acc:
            trainer.best_acc = val_results['accuracy']
            save_state(cp_manager, epoch, model, optimizer, float(trainer.best_acc))
            print("Waiting for final checkpoints to sync...")
            cp_manager.wait_until_finished()
            print("  ★ Checkpoint Saved")
    #jax.profiler.stop_trace()
