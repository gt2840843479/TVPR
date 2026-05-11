import math
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from models.mettack import BaseMeta
from criterion import *


class MetaDist(BaseMeta):
    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9, temperature=1, alpha=1):
        super(MetaDist, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()
        
    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(b.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
            if ix == len(self.weights) - 2:
                self.hidden = hidden

        logits = hidden
        output = F.log_softmax(logits, dim=1)

        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

        loss_logits_kd = rkl_loss(logits[idx_unlabeled], self.surrogate_logits[idx_unlabeled], temperature=self.temperature)
        # loss_logits_kd = kl_loss(logits[idx_unlabeled], self.surrogate_logits[idx_unlabeled], temperature=self.temperature)
        attack_loss = self.alpha * loss_logits_kd + (1 - self.alpha) * loss_labeled
            
        print('GCN loss on unlabeled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if i == 0:
                with torch.no_grad():
                    modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
                self.surrogate_weights = [w.detach().clone() for w in self.weights]
                self.surrogate_biases = [b.detach().clone() for b in self.biases]

                hidden = modified_features
                for ix, w in enumerate(self.surrogate_weights):
                    b = self.surrogate_biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = adj_norm @ hidden @ w + b
                    if self.with_relu and ix != len(self.surrogate_weights) - 1:
                        hidden = F.relu(hidden)
                    if ix == len(self.surrogate_weights) - 2:
                        self.surrogate_hidden = hidden
                logits = hidden
                self.surrogate_logits = logits
                output = F.log_softmax(logits, dim=1)
                labels_self_training = output.argmax(1)
                labels_self_training[idx_train] = labels[idx_train]

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
            
            if i >= 0:
                hidden = modified_features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)
                logits = hidden
                output = F.log_softmax(hidden, dim=1)

                loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

                weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
                self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
                if self.with_bias:
                    bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                    self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]
                
                with torch.no_grad():
                    self.surrogate_weights = [w - self.lr * v for w, v in zip(self.surrogate_weights, self.w_velocities)]
                    if self.with_bias:
                        self.surrogate_biases = [b - self.lr * v for b, v in zip(self.surrogate_biases, self.b_velocities)]

                    hidden = modified_features
                    for ix, w in enumerate(self.surrogate_weights):
                        b = self.surrogate_biases[ix] if self.with_bias else 0
                        if self.sparse_features:
                            hidden = adj_norm @ torch.spmm(hidden, w) + b
                        else:
                            hidden = adj_norm @ hidden @ w + b
                        if self.with_relu and ix != len(self.surrogate_weights) - 1:
                            hidden = F.relu(hidden)
                        if ix == len(self.surrogate_weights) - 2:
                            self.surrogate_hidden = hidden
                    logits = hidden
                    self.surrogate_logits = logits
                    output = F.log_softmax(logits, dim=1)
                    labels_self_training = output.argmax(1)
                    labels_self_training[idx_train] = labels[idx_train]

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)
        
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()
