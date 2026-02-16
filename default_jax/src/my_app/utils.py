from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path 

def setup_checkpointing(checkpoint_path: str):
    path = Path(checkpoint_path).absolute()
    
    # 1. Use StandardCheckpointHandler instead of StandardCheckpointer
    # 2. Pass it as the item_handlers (the second positional argument)
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
    # StandardSave remains the same for the newest versions
    manager.save(step, args=ocp.args.StandardSave(state))
