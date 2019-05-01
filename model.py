import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import args


# NN layers and models
class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        I = torch.eye(N).unsqueeze(0).to(args.device)
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        x_hat = []
        x_hat.append(torch.bmm(self.laplacian_batch(A), x))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers
        # print('out', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


class GCN(nn.Module):
    """
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                activation=nn.ReLU(inplace=True)) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x
