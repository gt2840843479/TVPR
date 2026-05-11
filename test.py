import torch
import numpy as np
from deeprobust.graph.utils import preprocess
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.defense import *
import argparse
import os
import yaml
import logging
from scipy.sparse import csr_matrix


def test(args, model, pyg_data, adj, features, labels, idx_train, idx_val, idx_test):
    if args.victim in ['GCN', 'GraphSAGE']:
        model.fit(features, adj, labels, idx_train)
        acc_test = model.test(idx_test)
    elif args.victim in ['GAT', 'SGC', 'MedianGCN']:
        model.fit(pyg_data)
        acc_test = model.test()
    elif args.victim == 'GCNJaccard':
        model.fit(features, adj, labels, idx_train, idx_val, threshold=0.01, verbose=False)
        acc_test = model.test(idx_test)
    elif args.victim == 'GCNSVD':
        model.fit(features, adj, labels, idx_train, idx_val, k=15, verbose=False)
        acc_test = model.test(idx_test)
    elif args.victim == 'ProGNN':
        model.fit(features, adj, labels, idx_train, idx_val)
        acc_test = model.test(features, labels, idx_test)
    elif args.victim in ['RGCN', 'SimPGCN']:
        model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=False)
        acc_test = model.test(idx_test)
    return acc_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
    parser.add_argument('--model', type=str, default='MetaDist',
            choices=['clean',
                     'random', 'DICE', 
                     'PGD_CE', 'PGD_CW',
                     'Meta-Self', 'Meta-Train', 'GraD', 'AtkSE', 'Metacon_S', 'Metacon_D', 
                     'MetaDist'], help='model variant')
    parser.add_argument('--victim', type=str, default='GCN',
            choices=['GCN', 'GraphSAGE', 'GAT', 'SGC', 'GCNJaccard', 'GCNSVD', 'RGCN', 'ProGNN', 'MedianGCN', 'SimPGCN'])
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=10)
    parser.add_argument('--graph_path', type=str, default='')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging_name = f'./logs/test_{args.victim}_{args.model}_{args.dataset}_{args.ptb_rate}.log'
    logging.basicConfig(
        level=logging.INFO,
        filename=logging_name,
        filemode='w',
        force=True
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type != 'cpu':
        torch.cuda.manual_seed(args.seed)

    data = Dataset(root='./data/', name=args.dataset, setting='nettack')
    
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if args.model != 'clean':
        graph_path = args.graph_path if args.graph_path else f'./ptb_graphs/{args.model}_{args.dataset}_{args.ptb_rate}.txt'
        adj = np.loadtxt(graph_path)
        adj = csr_matrix(adj)
        data.adj = adj

    if args.victim in ['RGCN', 'GCNSVD']:
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    elif args.victim in ['GCN', 'GAT', 'SGC', 'ProGNN', 'MedianGCN', 'SimPGCN']:
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
    
    pyg_data = Dpr2Pyg(data)

    if args.victim == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=16,
                    nclass=labels.max().item()+1,
                    dropout=0.5,
                    device=device)
    elif args.victim == 'GraphSAGE':
        from models.graphsage import GraphSAGE
        if args.dataset in ['citeseer', 'polblogs']:
            lr = 1e-3
        else:
            lr = 1e-2
        model = GraphSAGE(nfeat=features.shape[1],
                          nhid=16,
                          nclass=labels.max().item()+1,
                          dropout=0.5,
                          lr=lr,
                          device=device)
    elif args.victim == 'GAT':
        model = GAT(nfeat=features.shape[1],
                    nhid=8, heads=8,
                    nclass=labels.max().item()+1,
                    dropout=0.5,
                    device=device)
    elif args.victim == 'SGC':
        model = SGC(nfeat=features.shape[1],
                    nclass=labels.max().item()+1,
                    lr=0.1, 
                    device=device)
    elif args.victim == 'GCNJaccard':
        model = GCNJaccard(nfeat=features.shape[1], 
                           nclass=labels.max().item()+1,
                           nhid=16,
                           device=device)
    elif args.victim == 'GCNSVD':
        model = GCNSVD(nfeat=features.shape[1], 
                       nclass=labels.max().item()+1,
                       nhid=16, 
                       device=device)
    elif args.victim == 'MedianGCN':
        model = MedianGCN(nfeat=features.shape[1],
                          nhid=16,
                          nclass=labels.max().item()+1,
                          dropout=0.5,
                          device=device)
    elif args.victim == 'ProGNN':
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item()+1,
                  dropout=0.5, 
                  device=device)
        filename = os.path.join('configs/ProGNN.yaml')
        config = yaml.safe_load(open(filename).read())
        vars(args).update(config)
        model = ProGNN(gcn, args, device)
    elif args.victim == 'RGCN':
        model = RGCN(nnodes=adj.shape[0], 
                     nfeat=features.shape[1], 
                     nclass=labels.max().item()+1,
                     nhid=32, 
                     device=device)
    elif args.victim == 'SimPGCN':
        model = SimPGCN(nnodes=features.shape[0], 
                        nfeat=features.shape[1], 
                        nhid=16, 
                        nclass=labels.max().item()+1,
                        device=device)

    if args.victim != 'ProGNN':
        model = model.to(device)

    accs = []
    for i in range(args.runs):
        seed = args.start_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type != 'cpu':
            torch.cuda.manual_seed(seed)
        if args.victim in ['RGCN']:
            model._initialize()
        elif args.victim == 'ProGNN':
            model.model.initialize()
        else:
            model.initialize()
        acc = test(args, model, pyg_data, adj, features, labels, idx_train, idx_val, idx_test)
        accs.append(acc)

    print(f'test acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}')
    logger.info(f'test acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}')

if __name__ == '__main__':
    main()
