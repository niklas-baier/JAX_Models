# this is for visualization and logging only.
class TensorBoardLogger:
    def __init__(self,log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_weights(self, trainer, step, layer_name):
        trainer
def visualize_weights_of_layer(trainer, layer_name):
