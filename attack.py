import os
import yaml
import argparse

import torch
import numpy as np
from deeprobust.graph.utils import preprocess
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import Random, DICE, PGDAttack

from models.mettack import Metattack
from models.grad import GraD
from models.atkse import AtkSE
from models.metacon import Metacon_D, Metacon_S
from models.metadist import MetaDist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
    parser.add_argument('--model', type=str, default='MetaDist',
            choices=['random', 'DICE', 
                     'PGD_CE', 'PGD_CW',
                     'Meta-Self', 'Meta-Train', 'GraD', 'AtkSE', 'Metacon_S', 'Metacon_D', 
                     'MetaDist'], help='model variant')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type != 'cpu':
        torch.cuda.manual_seed(args.seed)

    os.makedirs('data', exist_ok=True)
    data = Dataset(root='./data/', name=args.dataset, setting='nettack')

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    perturbations = int(args.ptb_rate * (adj.sum()//2))

    if args.model not in ['random', 'DICE']:
        if args.model in ['PGD_CE', 'PGD_CW', 'Metacon_S', 'Metacon_D', 'Meta-Self', 'Meta-Train', 'GraD', 'MetaDist']:
            adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
        else:
            adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)

    if args.model == 'MetaDist':
        from models.gcn import GCN
    else:
        from deeprobust.graph.defense import GCN
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)

    if args.model in ['Meta-Self', 'GraD', 'Metacon_S', 'Metacon_D', 'MetaDist']:
        lambda_ = 0
    elif args.model in ['Meta-Train']:
        lambda_ = 1

    if args.model == 'GraD':
        if args.dataset == 'cora':
            momentum = 0.9
        elif args.dataset == 'cora_ml':
            momentum = 0.97
        elif args.dataset == 'citeseer':
            momentum = 0.9
        elif args.dataset == 'polblogs':
            momentum = 0.99
        elif args.dataset == 'pubmed':
            momentum = 0.98

    if args.model == 'random':
        model = Random()
    elif args.model == 'DICE':
        model = DICE()
    elif args.model == 'PGD_CE':
        model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device=device)
    elif args.model == 'PGD_CW':
        model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CW', device=device)
    elif args.model in ['Meta-Self', 'Meta-Train']:
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
    elif args.model == 'GraD':
        model = GraD(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_, momentum=momentum)
    elif args.model == 'MetaDist':
        filename = os.path.join('configs/MetaDist.yaml')
        config = yaml.safe_load(open(filename).read())[args.dataset]
        model = MetaDist(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_, momentum=config['momentum'], temperature=config['temperature'], alpha=config['alpha'], lr=config['lr'])
        model.edge_index = adj.nonzero().t().to(model.device)
    elif args.model == 'AtkSE':
        filename = os.path.join('configs/AtkSE.yaml')
        config = yaml.safe_load(open(filename).read())
        vars(args).update(config)
        model = AtkSE(args, nfeat=features.shape[1], hidden_sizes=[args.hidden], nnodes=adj.shape[0], nclass=labels.max().item()+1, dropout=0.5, train_iters=100, attack_features=False, lambda_=0, device=device, momentum=args.momentum)
    elif args.model == 'Metacon_S':
        filename = os.path.join('configs/Metacon_S.yaml')
        config = yaml.safe_load(open(filename).read())[args.dataset]
        model = Metacon_S(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_, 
                        analysis_mode=False, use_grad=config['use_grad'], train_iters=config['inner_train_iters'], lr=config['lr'], momentum=config['momentum'], 
                        droprate1=config['droprate1'], droprate2=config['droprate2'], coef1=config['coef1'], coef2=config['coef2'])
    elif args.model == 'Metacon_D':
        filename = os.path.join('configs/Metacon_D.yaml')
        config = yaml.safe_load(open(filename).read())[args.dataset]
        model = Metacon_D(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_, 
                        analysis_mode=False, train_iters=config['inner_train_iters'], lr=config['lr'], momentum=config['momentum'], droprate1=config['droprate1'], droprate2=config['droprate2'], 
                        coef1=config['coef1'], coef2=config['coef2'], vic_coef1=config['vic_coef1'], vic_coef2=config['vic_coef2'], vic_coef3=config['vic_coef3'])

    model = model.to(device)

    if args.model == 'random':
        model.attack(adj, perturbations)
        modified_adj = model.modified_adj
        modified_adj, _, _ = preprocess(modified_adj, features, labels, preprocess_adj=False)
    elif args.model == 'DICE':
        model.attack(adj, labels, perturbations)
        modified_adj = model.modified_adj
        modified_adj, _, _ = preprocess(modified_adj, features, labels, preprocess_adj=False)
    elif args.model in ['PGD_CE', 'PGD_CW']:
        fake_labels = surrogate.predict(features.to(device), adj.to(device))
        fake_labels = torch.argmax(fake_labels, 1).cpu()
        idx_fake = np.concatenate([idx_train, idx_test])
        idx_others = list(set(np.arange(len(labels))) - set(idx_train))
        fake_labels = torch.cat([labels[idx_train], fake_labels[idx_others]])
        model.attack(features, adj, fake_labels, idx_fake, perturbations, epochs=100)
        modified_adj = model.modified_adj
    elif args.model == 'AtkSE':
        modified_adj = model(features, adj, labels, idx_train, idx_unlabeled, perturbations)
        modified_adj = modified_adj.detach()
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj = model.modified_adj
    else:
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj = model.modified_adj

    os.makedirs('ptb_graphs', exist_ok=True) 
    np.savetxt(f'./ptb_graphs/{args.model}_{args.dataset}_{args.ptb_rate}.txt', modified_adj.cpu().numpy())

if __name__ == '__main__':
    main()
