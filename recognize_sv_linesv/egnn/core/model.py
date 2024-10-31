import torch.nn as nn
from core.egnn_clean import EGNN


class PL_EGNN(nn.Module):
    def __init__(self):
        super(PL_EGNN, self).__init__()
        self.model = EGNN(
            in_node_nf = 9,
            hidden_nf = 16,
            out_node_nf = 3,
            n_layers = 2,
            attention = True,
        )
        
    def forward(self, h, x, edges, return_features=False):
        h, x = self.model(h, x, edges, return_features=return_features)
        
        return h