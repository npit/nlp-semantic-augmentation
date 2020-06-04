import torch
from torch.nn import functional as F

from learning.neural import utils as neural_utils
from learning.neural.base_model import BaseModel
from learning.neural.dnn import SupervisedDNN


class MLPModel(BaseModel):
    """MLP """
    name = "mlp"
    wrapper_name = "supervised_dnn"

    def __init__(self, config, embeddings, output_dim):
        super(MLPModel, self).__init__(config, self.wrapper_name)

        self.config = config
        self.embedding_layer = neural_utils.make_embedding_layer(embeddings, config.train.train_embedding)

        hidden_dim = config.hidden_dim
        num_layers = config.num_layers
        self.linear_layers = neural_utils.make_linear_chain(embeddings.shape[-1], num_layers * [hidden_dim])
        # build final output layer
        self.linear_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input_data):
        """Forward pass method"""
        # embedding output
        if self.embedding_layer is not None:
            input_data = self.embedding_layer(input_data)
        # dense chain
        output = neural_utils.run_linear_chain(self.linear_layers, input_data)
        # for layer in self.linear_layers:
        #     data = F.dropout(F.relu(layer(data)), p=0.3)
        # output
        return self.linear_out(output)
