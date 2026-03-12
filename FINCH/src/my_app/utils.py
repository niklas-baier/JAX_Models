from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
import yaml

def setup_checkpointing(checkpoint_path: str):
    path = Path(checkpoint_path).absolute()

    checkpointer = ocp.CheckpointManager(
        path,
        item_handlers=ocp.StandardCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    )
    return checkpointer

def save_state(manager, step, model, optimizer, score):
    state = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
        'score': score
    }
    manager.save(step, args=ocp.args.StandardSave(state))

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
