"""
Reproduce TVPR results. All in one process.

Usage:
    python run_tvpr.py --model MetaDist --dataset cora --ptb_rate 0.05
    python run_tvpr.py --model Meta-Self --dataset cora --ptb_rate 0.05
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

from models.gcn import GCN
from models.metadist import MetaDist
from tvpr import tvpr_refine


def count_perturbations(ori_adj, current_adj):
    diff = (current_adj - ori_adj).abs() > 1e-12
    upper = torch.triu(diff, diagonal=1)
    return int(upper.sum().item())


def build_metadist(adj, features, labels, idx_train, dataset, device, sparse_features=False):
    surrogate = GCN(
        nfeat=features.shape[1], nclass=labels.max().item() + 1,
        nhid=16, dropout=0.5, with_relu=False, with_bias=True,
        weight_decay=5e-4, device=device,
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train)

    config = yaml.safe_load(Path("configs/MetaDist.yaml").read_text())[dataset]
    model = MetaDist(
        model=surrogate, nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True, attack_features=False,
        device=device, lambda_=0,
        momentum=config["momentum"], temperature=config["temperature"],
        alpha=config["alpha"], lr=config["lr"],
    ).to(device)
    model.edge_index = adj.nonzero().t().to(device)
    model.sparse_features = sparse_features
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MetaDist",
                        choices=["MetaDist", "Meta-Self", "Meta-Train"])
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--ptb_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type != "cpu":
        torch.cuda.manual_seed(args.seed)

    # Load data
    data = Dataset(root="./data/", name=args.dataset, setting="nettack")
    adj, features_sp, labels = data.adj, data.features, data.labels
    sparse_features = sp.issparse(features_sp)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    ori_adj, features, labels = preprocess(
        adj, features_sp, labels, preprocess_adj=False, device=device,
    )
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))

    # Step 1: Attack
    if args.model == "MetaDist":
        attack_model = build_metadist(ori_adj, features, labels, idx_train, args.dataset, device, sparse_features)
    else:
        from models.mettack import Metattack
        surrogate = GCN(
            nfeat=features.shape[1], nclass=labels.max().item() + 1,
            nhid=16, dropout=0.5, with_relu=False, with_bias=True,
            weight_decay=5e-4, device=device,
        ).to(device)
        surrogate.fit(features, ori_adj, labels, idx_train)
        attack_model = Metattack(
            model=surrogate, nnodes=ori_adj.shape[0],
            feature_shape=features.shape,
            attack_structure=True, attack_features=False,
            device=device, lambda_=0,
        ).to(device)
        attack_model.sparse_features = sparse_features

    print(f"{args.model} attack on {args.dataset}, budget={n_perturbations}")
    attack_model.attack(features, ori_adj, labels, idx_train, idx_unlabeled,
                        n_perturbations, ll_constraint=False)
    base_adj = attack_model.modified_adj.detach().clone()
    end_rng_state = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if device.type != "cpu" else None,
    }
    os.makedirs("ptb_graphs", exist_ok=True)
    np.savetxt(f"ptb_graphs/{args.model}_{args.dataset}_{args.ptb_rate}.txt",
               base_adj.cpu().numpy())

    # Step 2: TVPR refinement
    if args.model == "MetaDist":
        tvpr_model = attack_model  # reuse → RNG continuous → 62.07
    else:
        tvpr_model = build_metadist(ori_adj, features, labels, idx_train, args.dataset, device, sparse_features)
        np.random.set_state(end_rng_state["numpy"])
        torch.set_rng_state(end_rng_state["torch_cpu"])
        if device.type != "cpu" and end_rng_state["torch_cuda_all"] is not None:
            torch.cuda.set_rng_state_all(end_rng_state["torch_cuda_all"])
    t0 = time.time()
    refined_adj, stats = tvpr_refine(
        model=tvpr_model, features=features, ori_adj=ori_adj,
        labels=labels, idx_train=idx_train,
        idx_unlabeled=idx_unlabeled, base_adj=base_adj,
        n_perturbations=count_perturbations(ori_adj, base_adj),
        batch_size=args.batch_size, ll_constraint=False,
    )
    elapsed = time.time() - t0
    print(f"TVPR: {stats['steps']} rounds, {stats['swaps']} swaps, {elapsed:.1f}s")

    output_name = "MetaDist+TVPR" if args.model == "MetaDist" else "MetaAttack+TVPR"
    out_path = f"ptb_graphs/{output_name}_{args.dataset}_{args.ptb_rate}.txt"
    np.savetxt(out_path, refined_adj.cpu().numpy())

    # Step 3: Evaluate
    py = sys.executable
    eval_tag = "MetaDist" if args.model == "MetaDist" else args.model
    for victim in ["GCN", "GAT"]:
        r0 = subprocess.run([py, "test.py", "--model", eval_tag, "--dataset", args.dataset,
             "--ptb_rate", str(args.ptb_rate), "--victim", victim, "--runs", "10"],
            capture_output=True, text=True)
        r1 = subprocess.run([py, "test.py", "--model", eval_tag, "--dataset", args.dataset,
             "--ptb_rate", str(args.ptb_rate), "--victim", victim, "--runs", "10",
             "--graph_path", out_path], capture_output=True, text=True)
        for out_lines, label in [(r0.stdout, eval_tag), (r1.stdout, output_name)]:
            for line in out_lines.split("\n"):
                if "test acc" in line:
                    print(f"{label} {victim}: {line.strip()}")


if __name__ == "__main__":
    main()
