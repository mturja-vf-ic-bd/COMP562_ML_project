import torch
import torch.nn as nn
from torch.autograd import Variable
from settings import args


# NN layers and models
class SelectiveSequential(nn.Module):
    def __init__(self, module_dict):
        super(SelectiveSequential, self).__init__()
        for name, module in module_dict:
            self.add_module(name, module)

    def forward(self, x):
        out_list = []
        for name, module in self._modules.items():
            x = module(x)
            out_list.append(x[0])

        #stacked_output = torch.cat(out_list, dim=2)
        return out_list


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
        I = torch.eye(N).to(args.device)
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]

        x = torch.bmm(self.laplacian_batch(A), x)
        x = self.fc(x)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers
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
        layers = [('layer {}'.format(layer), GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                activation=nn.ReLU(inplace=True))) for layer, f in enumerate(filters)]
        self.gconv = SelectiveSequential(layers)

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(sum(filters), n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = sum(filters)
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        conv_x = self.gconv(data)
        if args.pool == 'max':
            x = torch.cat(conv_x, dim=2)
            x = torch.max(x, dim=1)[0].squeeze()
        elif args.pool == 'attentive':
            batch_size, n_nodes = conv_x[0].shape[:2]
            x = []
            # Node attention parameters for global pooling

            self.attn_param = Variable(torch.rand(1, n_nodes), requires_grad=True)
            for batch in conv_x:
                x.append(torch.mm(self.attn_param, batch.view(1, n_nodes, -1).squeeze()).view(batch_size, -1))

            x = torch.cat(x, dim=1)

        x = self.fc(x)
        return x
