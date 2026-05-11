import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml
from deeprobust.graph import utils
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

from criterion import rkl_loss
from models.gcn import GCN
from models.metadist import MetaDist


def forward_with_weights(model, features, adj_norm, weights, biases):
    hidden = features
    for ix, weight in enumerate(weights):
        bias = biases[ix] if model.with_bias else 0
        if model.sparse_features:
            hidden = adj_norm @ torch.spmm(hidden, weight) + bias
        else:
            hidden = adj_norm @ hidden @ weight + bias
        if model.with_relu and ix != len(weights) - 1:
            hidden = F.relu(hidden)
    return hidden


def sync_adj_changes(model, ori_adj, current_adj):
    with torch.no_grad():
        model.adj_changes.data = current_adj - ori_adj


def count_perturbations(ori_adj, current_adj):
    diff = (current_adj - ori_adj).abs() > 1e-12
    upper = torch.triu(diff, diagonal=1)
    return int(upper.sum().item())


def build_teacher_logits(model, features, ori_adj, labels, idx_train, idx_unlabeled):
    clean_norm = utils.normalize_adj_tensor(ori_adj)
    model.inner_train(features, clean_norm, idx_train, idx_unlabeled, labels)
    weights = [weight.detach().clone() for weight in model.weights]
    biases = [bias.detach().clone() for bias in model.biases]
    with torch.no_grad():
        teacher_logits = forward_with_weights(model, features, clean_norm, weights, biases).detach()
    return teacher_logits


def tvpr_refine(
    model,
    features,
    ori_adj,
    labels,
    idx_train,
    idx_unlabeled,
    base_adj,
    n_perturbations,
    batch_size,
    ll_constraint=False,
    ll_cutoff=0.004,
):
    current_adj = base_adj.detach().clone()
    sync_adj_changes(model, ori_adj, current_adj)

    row_idx, col_idx = torch.triu_indices(
        ori_adj.shape[0],
        ori_adj.shape[1],
        offset=1,
        device=model.device,
    )
    ori_upper = ori_adj[row_idx, col_idx]
    teacher_logits = build_teacher_logits(
        model,
        features,
        ori_adj,
        labels,
        idx_train,
        idx_unlabeled,
    )

    avg_upper_grad = None
    avg_count = 0
    accepted_steps = 0
    total_swaps = 0
    batch_size = max(1, int(batch_size))

    for _ in range(max(1, int(n_perturbations))):
        work_adj = current_adj.detach().clone().requires_grad_(True)
        adj_norm = utils.normalize_adj_tensor(work_adj)

        model.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
        logits = forward_with_weights(model, features, adj_norm, model.weights, model.biases)
        output = F.log_softmax(logits, dim=1)
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_kd = rkl_loss(
            logits[idx_unlabeled],
            teacher_logits[idx_unlabeled],
            temperature=model.temperature,
        )
        attack_loss = model.alpha * loss_kd + (1 - model.alpha) * loss_labeled
        adj_grad = torch.autograd.grad(attack_loss, work_adj)[0].detach()

        if model.undirected:
            upper_grad = adj_grad[row_idx, col_idx] + adj_grad[col_idx, row_idx]
        else:
            upper_grad = adj_grad[row_idx, col_idx]

        if avg_upper_grad is None:
            avg_upper_grad = upper_grad
            avg_count = 1
        else:
            avg_upper_grad = (avg_upper_grad * avg_count + upper_grad) / (avg_count + 1)
            avg_count += 1

        current_upper = current_adj[row_idx, col_idx]
        held_mask = (current_upper - ori_upper).abs() > 1e-12
        if int(held_mask.sum().item()) == 0:
            break

        keep_direction = (current_upper - ori_upper).sign()
        add_direction = 1.0 - 2.0 * current_upper
        keep_values = avg_upper_grad * keep_direction
        add_values = avg_upper_grad * add_direction

        singleton_allowed = model.filter_potential_singletons(current_adj)[row_idx, col_idx].detach() > 0
        can_revert_held = (keep_direction < 0) | singleton_allowed
        legal_candidates = (~held_mask) & torch.isfinite(add_values)
        legal_candidates &= (current_upper < 0.5) | singleton_allowed

        if ll_constraint:
            allowed_mask, _ = model.log_likelihood_constraint(current_adj, ori_adj, ll_cutoff)
            legal_candidates &= allowed_mask.to(model.device)[row_idx, col_idx] > 0

        held_positions = torch.nonzero(
            held_mask & can_revert_held & torch.isfinite(keep_values),
            as_tuple=False,
        ).view(-1)
        candidate_positions = torch.nonzero(legal_candidates, as_tuple=False).view(-1)
        if held_positions.numel() == 0 or candidate_positions.numel() == 0:
            break

        sorted_held = held_positions[torch.argsort(keep_values[held_positions], descending=False)]
        sorted_candidates = candidate_positions[
            torch.argsort(add_values[candidate_positions], descending=True)
        ]
        pair_count = min(batch_size, int(sorted_held.numel()), int(sorted_candidates.numel()))
        margin = torch.median(torch.abs(keep_values[held_positions])).item()

        swaps = []
        for pair_idx in range(pair_count):
            worst_pos = int(sorted_held[pair_idx].item())
            best_pos = int(sorted_candidates[pair_idx].item())
            gain = (add_values[best_pos] - keep_values[worst_pos]).item()
            if gain <= margin:
                break
            swaps.append((worst_pos, best_pos))

        if not swaps:
            break

        for worst_pos, best_pos in swaps:
            old_row = int(row_idx[worst_pos].item())
            old_col = int(col_idx[worst_pos].item())
            new_row = int(row_idx[best_pos].item())
            new_col = int(col_idx[best_pos].item())

            current_adj[old_row, old_col] = ori_adj[old_row, old_col]
            current_adj[old_col, old_row] = ori_adj[old_col, old_row]
            current_adj[new_row, new_col] = 1.0 - current_adj[new_row, new_col]
            current_adj[new_col, new_row] = current_adj[new_row, new_col]

        accepted_steps += 1
        total_swaps += len(swaps)

    sync_adj_changes(model, ori_adj, current_adj)
    return current_adj.detach(), {
        "steps": accepted_steps,
        "swaps": total_swaps,
        "gradient_evaluations": avg_count,
    }


def default_output_name(raw_model):
    mapping = {
        "MetaDist": "MetaDist_TVPR",
        "Meta-Self": "MetaAttackSelf_TVPR",
        "Meta-Train": "MetaAttackTrain_TVPR",
    }
    return mapping.get(raw_model, raw_model.replace("-", "") + "_TVPR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "cora_ml", "citeseer", "polblogs"],
    )
    parser.add_argument("--ptb_rate", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--raw_model", type=str, default="MetaDist")
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--output_model", type=str, default=None)
    parser.add_argument("--ll_constraint", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type != "cpu":
        torch.cuda.manual_seed(args.seed)

    data = Dataset(root="./data/", name=args.dataset, setting="nettack")
    adj, features, labels = data.adj, data.features, data.labels
    sparse_features = sp.issparse(features)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(
        adj,
        features,
        labels,
        preprocess_adj=False,
        device=device,
    )

    raw_path = args.raw_path or os.path.join(
        "ptb_graphs",
        f"{args.raw_model}_{args.dataset}_{args.ptb_rate}.txt",
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Initial poisoned graph not found: {raw_path}. "
            f"Run attack.py first or pass --raw_path."
        )
    base_adj = torch.FloatTensor(np.loadtxt(raw_path)).to(device)
    n_perturbations = count_perturbations(adj, base_adj)

    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train)

    config = yaml.safe_load(open(os.path.join("configs", "MetaDist.yaml")).read())[args.dataset]
    model = MetaDist(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=0,
        momentum=config["momentum"],
        temperature=config["temperature"],
        alpha=config["alpha"],
        lr=config["lr"],
    ).to(device)
    model.edge_index = adj.nonzero().t().to(model.device)
    model.sparse_features = sparse_features

    start = time.perf_counter()
    refined_adj, stats = tvpr_refine(
        model=model,
        features=features,
        ori_adj=adj,
        labels=labels,
        idx_train=idx_train,
        idx_unlabeled=idx_unlabeled,
        base_adj=base_adj,
        n_perturbations=n_perturbations,
        batch_size=args.batch_size,
        ll_constraint=args.ll_constraint,
    )
    elapsed = time.perf_counter() - start

    output_model = args.output_model or default_output_name(args.raw_model)
    os.makedirs("ptb_graphs", exist_ok=True)
    out_path = os.path.join(
        "ptb_graphs",
        f"{output_model}_{args.dataset}_{args.ptb_rate}.txt",
    )
    np.savetxt(out_path, refined_adj.cpu().numpy())

    print(
        "dataset={},ptb_rate={},raw_model={},output_model={},budget={},batch_size={},steps={},swaps={},seconds={:.3f},output={}".format(
            args.dataset,
            args.ptb_rate,
            args.raw_model,
            output_model,
            n_perturbations,
            args.batch_size,
            stats["steps"],
            stats["swaps"],
            elapsed,
            out_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
