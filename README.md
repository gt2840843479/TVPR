# TVPR — Temporal-Average-Value-Guided Perturbation Refinement

Code for reproducing **Table 2** results of the paper. Built upon the official [MetaDist](https://github.com/Crysta1ovo/MetaDist) implementation (AAAI 2026).

## Requirements

```bash
pip install torch torch-geometric deeprobust numpy scipy pyyaml
```

Python 3.8+, CUDA 11.8+, PyTorch 2.4+ recommended.

## Reproduce

### TVPR (ours)

```bash
cd TVPR

# MetaDist + TVPR (recommended, batch_size default=4)
python run_tvpr.py --model MetaDist --dataset cora --ptb_rate 0.05

# Meta-Self (MetaAttack) + TVPR
python run_tvpr.py --model Meta-Self --dataset cora --ptb_rate 0.05

# Custom batch size
python run_tvpr.py --model MetaDist --dataset cora --ptb_rate 0.05 --batch_size 4
```

### Baselines

```bash
# MetaDist attack + evaluate
python attack.py --model MetaDist --dataset cora --ptb_rate 0.05
python test.py --model MetaDist --dataset cora --victim GCN --ptb_rate 0.05
python test.py --model MetaDist --dataset cora --victim GAT --ptb_rate 0.05

# Meta-Self, Meta-Train
python attack.py --model Meta-Train --dataset cora --ptb_rate 0.05
python attack.py --model Meta-Self --dataset cora --ptb_rate 0.05
python test.py --model Meta-Train --dataset cora --victim GCN --ptb_rate 0.05
python test.py --model Meta-Self --dataset cora --victim GCN --ptb_rate 0.05

# AtkSE, GraD, Metacon
python attack.py --model AtkSE --dataset cora --ptb_rate 0.05
python attack.py --model GraD --dataset cora --ptb_rate 0.05
python attack.py --model Metacon_S --dataset cora --ptb_rate 0.05
python attack.py --model Metacon_D --dataset cora --ptb_rate 0.05

# PGD-based attacks
python attack.py --model PGD_CE --dataset cora --ptb_rate 0.05
python attack.py --model PGD_CW --dataset cora --ptb_rate 0.05

# Random baselines
python attack.py --model Random --dataset cora --ptb_rate 0.05
python attack.py --model DICE --dataset cora --ptb_rate 0.05
```

### Evaluate any poisoned graph

```bash
python test.py --model MetaDist --dataset cora --victim GCN --ptb_rate 0.05 \
    --graph_path ptb_graphs/MetaDist+TVPR_cora_0.05.txt
```

## Verified Results (Cora, 5% perturbation)

| Method | GCN | GAT |
|--------|-----|-----|
| MetaDist | 67.23±1.21 | 78.47±0.35 |
| **MetaDist+TVPR (Ours)** | **62.07±1.86** | **76.88±1.05** |
| **Meta-Self+TVPR (Ours)** | **66.66±0.43** | **76.94±1.13** |

Pre-verified graphs are included at `ptb_graphs/MetaDist+TVPR_cora_0.05.txt`.

## Files

| File | Source | Description |
|------|--------|-------------|
| `tvpr.py` | **This paper** | TVPR refinement algorithm |
| `run_tvpr.py` | **This paper** | Attack → TVPR → evaluate pipeline |
| `test.py` | MetaDist (+ `--graph_path`) | Evaluation |
| `attack.py` | MetaDist | Attack interface |
| `models/` | MetaDist | GCN, MetaDist, Meta-Self, baselines |
| `configs/` | MetaDist | Attack configurations |
| `data/` | MetaDist | Cora, CiteSeer, PolBlogs, Cora-ML |

## Citation

```bibtex
@article{yao2026tvpr,
  title={Does Sequential Greedy Graph Poisoning Exhibit Perturbation Value Deviation?
         Temporal-Average-Value-Guided Perturbation Refinement},
  author={Yao, Xing and Liu, Hai and Liu, Zhiquan},
  journal={arXiv preprint},
  year={2026}
}

@inproceedings{peng2026metadist,
  title={Surrogate as Teacher: Distillation-Guided Graph Poisoning Attack},
  author={Peng, Xingyu and Xu, Ke},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
