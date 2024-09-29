# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:13:21 2020
@author: Hristo Petkov
"""

"""
Modifications copyright (C) 2021 Hristo Petkov
Modifications are as follows:
  -Addition of new utility functions which works with the new modifications in the other files
"""

"""
@inproceedings{yu2019dag,
  title={DAG-GNN: DAG Structure Learning with Graph Neural Networks},
  author={Yue Yu, Jie Chen, Tian Gao, and Mo Yu},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}

@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
"""
import torch
import os
import math
import numpy as np
import torch.nn as nn
import networkx as nx
import scipy.linalg as slin
import torch.nn.functional as F
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from networkx.convert_matrix import from_numpy_matrix
from matplotlib import pyplot as plt
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from FullDataPreProcessor import FullDataPreProcessor

# AAE utility functions

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def preprocess_adj_new(adj, device):
    adj_normalized = (torch.eye(adj.shape[0]).double().to(device) - (adj.transpose(0,1)).to(device))
    return adj_normalized

def preprocess_adj_new1(adj, device):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double().to(device) - adj.transpose(0,1).to(device))
    return adj_normalized

def matrix_poly(matrix, d, device):
    x = torch.eye(d).double().to(device) + torch.div(matrix, d).to(device)
    return torch.matrix_power(x, d)

# compute constraint h(A) value
def _h_A(A, m, device):
    expm_A = matrix_poly(A*A, m, device)
    h_A = torch.trace(expm_A) - m
    return h_A

def nll_catogrical(preds, target, add_const = False, eps=1e-16):
    '''compute the loglikelihood of discrete variables
    '''
    loss = nn.CrossEntropyLoss(reduction='sum')
    output = loss(preds, torch.argmax(target, 1))
    return output     

def nll_gaussian(preds, target, variance, add_const=False):
    
    mean1 = preds
    mean2 = target
    
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
            
    return neg_log_p.sum() / (target.size(0))
    
def kl_gaussian_sem(logits):
    
    mu = logits
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (logits.size(0)))*0.5

    #if loss_type = 'H':
        #kl_sum = beta * kld_sum
        #return (kl_sum / (logits.size(0)))*0.5
    #elif loss_type = 'B':
        #C_max = C_max.to(device)
        #C = torch.clamp(C_max/C_stop_iter * num_iter, 0, C_max.data[0])
        #kl_sum = sigma*(kl_sum-C).abs()
        #return (kl_sum / (logits.size(0)))*0.5
    #else:
        #return (kl_sum / (logits.size(0)))*0.5
          
def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))
     

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise).double()
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

#Plotting the DAG
#Borrowed from Causalnex Documentation
#https://causalnex.readthedocs.io/en/latest/03_tutorial/plotting_tutorial.html
def draw_dag(graph, data_type, columns = None):
    
    final_DAG = from_numpy_matrix(graph, create_using=nx.DiGraph)
    
    if data_type == 'real':
        final_DAG = nx.relabel_nodes(
            final_DAG, dict(zip(list(range(graph.shape[0])), columns)))
    final_DAG.remove_nodes_from(list(nx.isolates(final_DAG)))
    
    print('FINAL DAG')
    print(final_DAG.adj)
    
    write_dot(final_DAG,'test.dot')
    
    fig = plt.figure(figsize=(15, 8))  # set figsize
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#001521")  # set backgrount

    pos = graphviz_layout(final_DAG, prog="dot")

    # add nodes to figure
    nx.draw_networkx_nodes(
        final_DAG,
        pos,
        node_shape="H",
        node_size=1000,
        linewidths=3,
        edgecolors="#4a90e2d9",
    )
    
    # add labels
    nx.draw_networkx_labels(
        final_DAG,
        pos,
        font_color="#FFFFFFD9",
        font_weight="bold",
        font_family="Helvetica",
        font_size=10,
    )
    
    # add edges
    nx.draw_networkx_edges(
        final_DAG,
        pos,
        edge_color="white",
        node_shape="H",
        node_size=2000,
        width=[w + 0.1 for _, _, w, in final_DAG.edges(data="weight")],
    )

    plt.show()
    plt.close()
    
# data generating functions below this point

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.
    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)
    Returns:
        G: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G

def simulate_sem(G: nx.DiGraph,
                 n: int, x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.
    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
    Returns:
        X: [n,d] sample matrix
    """
    
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        elif linear_type == 'post_nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'post_nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'post_nonlinear_1':
                X[:, j, 0] = np.tanh(eta + np.random.normal(scale=noise_scale, size=n))
            elif linear_type == 'post_nonlinear_2':
                X[:, j, 0] = np.tanh(2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n))
        elif sem_type == 'linear-exp':
            X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X

def simulate_population_sample(W: np.ndarray,
                               Omega: np.ndarray) -> np.ndarray:
    """Simulate data matrix X that matches population least squares.
    Args:
        W: [d,d] adjacency matrix
        Omega: [d,d] noise covariance matrix
    Returns:
        X: [d,d] sample matrix
    """
    d = W.shape[0]
    X = np.sqrt(d) * slin.sqrtm(Omega).dot(np.linalg.pinv(np.eye(d) - W))
    return X

def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.
    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def compute_BiCScore(G, D):
    '''compute the bic score'''
    # score = gm.estimators.BicScore(self.data).score(self.model)
    origin_score = []
    num_var = G.shape[0]
    for i in range(num_var):
        parents = np.where(G[:,i] !=0)
        score_one = compute_local_BiCScore(D, i, parents)
        origin_score.append(score_one)

    score = sum(origin_score)

    return score


def compute_local_BiCScore(np_data, target, parents):
    # use dictionary
    sample_size = np_data.shape[0]
    var_size = np_data.shape[1]

    # build dictionary and populate
    count_d = dict()
    if len(parents) < 1:
        a = 1

    # unique_rows = np.unique(self.np_data, axis=0)
    # for data_ind in range(unique_rows.shape[0]):
    #     parent_combination = tuple(unique_rows[data_ind,:].reshape(1,-1)[0])
    #     count_d[parent_combination] = dict()
    #
    #     # build children
    #     self_value = tuple(self.np_data[data_ind, target].reshape(1,-1)[0])
    #     if parent_combination in count_d:
    #         if self_value in count_d[parent_combination]:
    #             count_d[parent_combination][self_value] += 1.0
    #         else:
    #             count_d[parent_combination][self_value] = 1.0
    #     else:
    #         count_d[parent_combination] = dict()
    #         count_d

    # slower implementation
    for data_ind in range(sample_size):
        parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
        self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
        if parent_combination in count_d:
            if self_value in count_d[parent_combination]:
                count_d[parent_combination][self_value] += 1.0
            else:
                count_d[parent_combination][self_value] = 1.0
        else:
            count_d[parent_combination] = dict()
            count_d[parent_combination][self_value] = 1.0

    # compute likelihood
    loglik = 0.0
    # for data_ind in range(sample_size):
    # if len(parents) > 0:
    num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
    # else:
    #    num_parent_state = 0
    num_self_state = np.amax(np_data[:, target], axis=0) + 1

    for parents_state in count_d:
        local_count = sum(count_d[parents_state].values())
        for self_state in count_d[parents_state]:
            loglik += count_d[parents_state][self_state] * (
                        math.log(count_d[parents_state][self_state] + 0.1) - math.log(local_count))

    # penality
    num_param = num_parent_state * (
                num_self_state - 1)  # count_faster(count_d) - len(count_d) - 1 # minus top level and minus one
    bic = loglik - 0.5 * math.log(sample_size) * num_param

    return bic

def data_to_tensor_dataset(X, batch_size, G=None):
        
    feat_train = torch.FloatTensor(X)
    train_data = TensorDataset(feat_train, feat_train)
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    
    return train_data_loader, G

def load_data(args, batch_size=1000, suffix='', debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type, linear_type = args.graph_type, args.graph_degree, args.graph_sem_type, args.graph_linear_type
    x_dims = args.x_dims

    if args.data_type == 'synthetic':
        # generate data
        G = simulate_random_dag(d, degree, graph_type)
        X = simulate_sem(G, n, x_dims, sem_type, linear_type)
        
        train_data_loader, G = data_to_tensor_dataset(X, batch_size, G)
        return train_data_loader, G

    elif args.data_type == 'real':
        #this where you can use your own dataset
        assert args.path != '', 'Data path must be specified'
        fdpp = FullDataPreProcessor(args.path, args.column_names_list, args.initial_identifier, args.num_of_rows, args.seed)
        preprocessed_dataframe = fdpp.get_dataframe()
        columns = fdpp.sample_dataframe(preprocessed_dataframe[0]).columns
        X = fdpp.sample_dataframe(preprocessed_dataframe[0]).values
                
        train_data_loader, G = data_to_tensor_dataset(X, batch_size)
        return train_data_loader, X.shape[1], columns
    
    elif args.data_type == 'benchmark':
        # create your own version of benchmark discrete data
        assert args.path != '', 'Data path must be specified'
        file_path_dataset = os.path.join(args.path, 'enter data file here') #e.g for pathfinder benchmark dataset it should be something like pathfinder_5000.txt
        
        # read file
        data = np.loadtxt(file_path_dataset, skiprows =0, dtype=np.int32)
            
        # read ground truth graph
        file_path = os.path.join(args.path, 'enter ground truth file here') #e.g for pathfinder benchmark dataset it should be somethiing like pathfinder_graph.txt
        
        graph = np.loadtxt(file_path, skiprows =0, dtype=np.int32)
            
        G = nx.DiGraph(graph)
        X = data[:args.num_of_rows]
        
        train_data_loader, G = data_to_tensor_dataset(X, batch_size, G)

        return train_data_loader, X.shape[1], G
    
