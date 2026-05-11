import math
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from torch_geometric.utils import dense_to_sparse, to_dense_adj, degree, to_undirected
from models.mettack import BaseMeta


class Metacon_S(BaseMeta):
    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, 
    droprate1=0.5, droprate2=0.5, coef1=1.0, coef2=0.01,
    train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False, use_grad=False):

        super(Metacon_S, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.momentum = momentum
        self.lr = lr

        self.droprate1 = droprate1
        self.droprate2 = droprate2
        self.coef1 = coef1
        self.coef2 = coef2

        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode
        self.use_grad = use_grad

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
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)


    def drop_edge_weighted(self, edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        return edge_index[:, sel_mask]

    def drop_edge(self, edge_index, p):
        
        return self.drop_edge_weighted(edge_index, self.drop_weights, p=p, threshold=0.7)
    
    def degree_drop_weights(self, edge_index):
        edge_index_ = to_undirected(edge_index)
        deg = degree(edge_index_[1])
        deg_col = deg[edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        return weights
        
    def inner_train(self, features, adj_norm, adj, idx_train, idx_unlabeled, labels):
        
        self._initialize()

        edge_index = dense_to_sparse(adj)[0]

        self.drop_weights = self.degree_drop_weights(edge_index).to(self.device)

        # edge_index1 = self.drop_edge(edge_index, self.droprate1)
        edge_index2 = self.drop_edge(edge_index, self.droprate2)

        # adj_norm1 = utils.normalize_adj_tensor(to_dense_adj(edge_index1, max_num_nodes=self.nnodes)[0])
        # adj_norm1 = adj_norm
        adj_norm2 = utils.normalize_adj_tensor(to_dense_adj(edge_index2, max_num_nodes=self.nnodes)[0])

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

            hidden1 = features
            hidden2 = features
            for ix, w in enumerate(self.weights):
                
                b = self.biases[ix] if self.with_bias else 0

                if self.with_relu and ix != len(self.weights) - 1:
                    continue
                
                if self.sparse_features:
                    hidden1 = adj_norm @ torch.spmm(hidden1, w) + b
                    hidden2 = adj_norm2 @ torch.spmm(hidden2, w) + b
                else:
                    hidden1 = adj_norm @ hidden1 @ w + b
                    hidden2 = adj_norm2 @ hidden2 @ w + b

                

            output = F.log_softmax(hidden, dim=1)

            output1 = F.elu(hidden1)
            output2 = F.elu(hidden2)

            # loss_labeled = (F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train]))/2
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train]) 
            
            self.tau = 0.5
            f = lambda x: torch.exp(x / self.tau)

            # output1 = F.normalize(output1)
            # output2 = F.normalize(output2)
            
            refl = f(torch.mm(output1, output1.t()))
            between = f(torch.mm(output1, output2.t()))
            loss_unlabeled = -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))

            # loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled[idx_unlabeled].mean()
            loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled[idx_unlabeled].mean()

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
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

        output = F.log_softmax(hidden, dim=1)
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        if self.use_grad:
            p = torch.exp(output)
            loss_unlabeled = -p[idx_unlabeled, labels_self_training[idx_unlabeled]].mean()
        else:
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad


    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):

        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        attacked_idx = []
        self.analysis_idx = []
        self.analysis_inner_grad = []
        self.analysis_output = []


        self.set_grad_stats()

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_features, adj_norm, modified_adj, idx_train, idx_unlabeled, labels)

            

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            with torch.no_grad():
                topk = 30
                adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
                topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
                topk_grad = adj_grad_flat[topk_index]
                topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

                topk_ll, topk_ul, topk_uu = [], [], []

                for j, (row, col) in enumerate(topk_index):
                    if (row in idx_train) & (col in idx_train):
                        topk_ll.append(topk_grad[j].item())
                    elif (row in idx_train) & (col not in idx_train):
                        topk_ul.append(topk_grad[j].item())
                    elif (row not in idx_train) & (col not in idx_train):
                        topk_uu.append(topk_grad[j].item())

                self.grad_stats['grad_topk_avg_ll'].append(np.mean(topk_ll))
                self.grad_stats['grad_topk_avg_ul'].append(np.mean(topk_ul))
                self.grad_stats['grad_topk_avg_uu'].append(np.mean(topk_uu))
                self.grad_stats['grad_topk_cnt_ll'].append(len(topk_ll))
                self.grad_stats['grad_topk_cnt_ul'].append(len(topk_ul))
                self.grad_stats['grad_topk_cnt_uu'].append(len(topk_uu))

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

        
            adj_meta_argmax = torch.argmax(adj_meta_score)
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if self.undirected:
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)


            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}


class Metacon_D(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, 
    droprate1=0.5, droprate2=0.5, coef1=1.0, coef2=0.01, vic_coef1=1.0, vic_coef2=1.0, vic_coef3=0.04,
    train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(Metacon_D, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.momentum = momentum
        self.lr = lr

        self.droprate1 = droprate1
        self.droprate2 = droprate2
        self.coef1 = coef1
        self.coef2 = coef2

        self.vic_coef1 = vic_coef1
        self.vic_coef2 = vic_coef2
        self.vic_coef3 = vic_coef3

        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode

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
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def drop_edge_weighted(self, edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        return edge_index[:, sel_mask]

    def drop_edge(self, edge_index, p):
        
        return self.drop_edge_weighted(edge_index, self.drop_weights, p=p, threshold=0.7)
    
    def degree_drop_weights(self, edge_index):
        edge_index_ = to_undirected(edge_index)
        deg = degree(edge_index_[1])
        deg_col = deg[edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        return weights
        
    def inner_train(self, features, adj_norm, adj, idx_train, idx_unlabeled, labels):
        
        self._initialize()

        edge_index = dense_to_sparse(adj)[0]

        self.drop_weights = self.degree_drop_weights(edge_index).to(self.device)

        # edge_index1 = self.drop_edge(edge_index, self.droprate1)
        edge_index2 = self.drop_edge(edge_index, self.droprate2)

        # adj_norm1 = utils.normalize_adj_tensor(to_dense_adj(edge_index1, max_num_nodes=self.nnodes)[0])
        # adj_norm1 = adj_norm
        adj_norm2 = utils.normalize_adj_tensor(to_dense_adj(edge_index2, max_num_nodes=self.nnodes)[0])

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
            hidden2 = features
            
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
            
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                    hidden2 = adj_norm2 @ torch.spmm(hidden2, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                    hidden2 = adj_norm2 @ hidden2 @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)
                
                if ix == len(self.weights) - 1:
                    z1 = F.elu(hidden)
                    z2 = F.elu(hidden2)


            output = F.log_softmax(hidden, dim=1)

            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train]) 

            ## VIC REG
            bs, num_feat = z1.shape

            z1 = z1[idx_unlabeled]
            z2 = z2[idx_unlabeled]

            # mse loss
            mse_loss = F.mse_loss(z1, z2)

            # std loss
            z1 = z1 - z1.mean(dim=0)
            z2 = z2 - z2.mean(dim=0)
            std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4) 
            std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4) 
            std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
            
            # cov loss
            cov_z1 = (z1.T @ z1) / (bs - 1)
            cov_z2 = (z2.T @ z2) / (bs - 1)
            cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(num_feat) + off_diagonal(cov_z2).pow_(2).sum().div(num_feat)

            loss_unlabeled = self.vic_coef1 * mse_loss + self.vic_coef2 * std_loss + self.vic_coef3 * cov_loss


            loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]


        print(f"Labeled Loss:{loss_labeled.item()} and Unlabeled Loss:{loss_unlabeled.item()}")
    
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

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):

        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        # labels_self_training = self.self_training_softlabel(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        attacked_idx = []
        self.analysis_idx = []
        self.analysis_inner_grad = []
        self.analysis_output = []


        self.set_grad_stats()

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_features, adj_norm, modified_adj, idx_train, idx_unlabeled, labels)

            

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            with torch.no_grad():
                topk = 30
                adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
                topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
                topk_grad = adj_grad_flat[topk_index]
                topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

                topk_ll, topk_ul, topk_uu = [], [], []

                for j, (row, col) in enumerate(topk_index):
                    if (row in idx_train) & (col in idx_train):
                        topk_ll.append(topk_grad[j].item())
                    elif (row in idx_train) & (col not in idx_train):
                        topk_ul.append(topk_grad[j].item())
                    elif (row not in idx_train) & (col not in idx_train):
                        topk_uu.append(topk_grad[j].item())

                self.grad_stats['grad_topk_avg_ll'].append(np.mean(topk_ll))
                self.grad_stats['grad_topk_avg_ul'].append(np.mean(topk_ul))
                self.grad_stats['grad_topk_avg_uu'].append(np.mean(topk_uu))
                self.grad_stats['grad_topk_cnt_ll'].append(len(topk_ll))
                self.grad_stats['grad_topk_cnt_ul'].append(len(topk_ul))
                self.grad_stats['grad_topk_cnt_uu'].append(len(topk_uu))

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

        
            adj_meta_argmax = torch.argmax(adj_meta_score)
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if self.undirected:
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)


            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()