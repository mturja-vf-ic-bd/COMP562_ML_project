import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.datasets import TUDataset
from settings import args
import numpy as np
from train_test import train, test
from utils import split_ids, collate_batch
from model import GCN


def setup():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    setup()
    rnd_state = np.random.RandomState(args.seed)

    print("Loading data ... ... ...")
    dataset = TUDataset('./data/%s/' % args.dataset, name=args.dataset,
                        use_node_attr=args.use_cont_node_attr)
    train_ids, test_ids = split_ids(rnd_state.permutation(len(dataset)), folds=args.n_folds)
    print("Data loaded !!!")

    acc_folds = []
    for fold_id in range(args.n_folds):
        loaders = []
        for split in ['train', 'test']:
            gdata = dataset[torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id])]
            loader = DataLoader(gdata,
                                batch_size=args.batch_size,
                                shuffle=split.find('train') >= 0,
                                num_workers=args.threads,
                                collate_fn=collate_batch)
            loaders.append(loader)

        print('\nFOLD {}, train {}, test {}'.format(fold_id, len(loaders[0].dataset), len(loaders[1].dataset)))

        model = GCN(in_features=loaders[0].dataset.num_features,
                    out_features=loaders[0].dataset.num_classes,
                    n_hidden=args.n_hidden,
                    filters=args.filters,
                    dropout=args.dropout).to(args.device)
        print('\nInitialize model')
        print(model)
        train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999))
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
        for epoch in range(args.epochs):
            train(loaders[0], model=model, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            acc = test(loaders[1], model=model, epoch=epoch)
        acc_folds.append(acc)

    print(acc_folds)
    print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, np.mean(acc_folds), np.std(acc_folds)))


if __name__ == '__main__':
    main()
