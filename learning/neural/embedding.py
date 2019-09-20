from torch.nn import Embedding

from learning.learner import Learner

"""Base neural model with learnable input embeddings"""
# class Embedding(Learner, NeuralModel):
class EmbeddingNN(Learner):
    name = "embedding"
    def __init__(self, config):
        self.config = config
        super().__init__()

    def make(self):
        super().make()

    def train_model(self, train_data, train_labels, val_data, val_labels):
        pass

# torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
