"""
Collection of methods used across files.
The data generation is a modified version from Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. See https://github.com/xunzheng/notears/blob/master/LICENSE for their license.
"""
from scipy.special import expit as sigmoid
from collections import namedtuple
from pathlib import Path
from datetime import datetime
import igraph as ig
import numpy as np
import pandas as pd
import pickle as pk
import os
import shutil
import random
import torch


def thresholds(x):
    """ use x as threshold for all algorithms that require thresholding"""
    return {"lingamIC":                   -np.inf,
            "fges":                       'bidirected',  # favourable
            "pc":                         'bidirected',  # favourable
            "randomregressIC":            -np.inf,
            "sortnregressIC":             -np.inf,
            "notearsLinear":              x,
            "golemEV_orig":               x,
            "golemNV_orig":               x,
            "golemEV_golemNV_orig":       x,
            "empty":                      -np.inf}

def special_thres():
    return [
        "notearsLinear",
        "golemEV_orig",
        "golemNV_orig",
        "golemEV",
        "golemNV",
    ]

# Experiment description
DatasetDescription = namedtuple("DatasetDescription",
                                ["graph",
                                "noise",
                                "noise_variance",
                                "edge_weight_range",
                                "n_nodes",
                                "n_obs",
                                "random_seed"])

# Actual experiment
dataset_fields = ["description",
                  "W_true",
                  "B_true",
                  "data",
                  "hash",
                  "scaler",
                  "scaling_factors",
                  "varsortability"]
Dataset = namedtuple("Dataset", dataset_fields)
Dataset.__new__.__defaults__ = (None,) * len(dataset_fields)

# noise distribution dtype
NoiseDistribution = namedtuple("NoiseDistribution", ["type", "uniform_variance"])


def matching_dataset(hash, datasets):
    """ get dataset with same hash from list of datasets """
    idx = [dataset.hash for dataset in datasets].index(hash)
    return datasets[idx]


def l2_loss(X, W):
    return 0.5 / X.shape[0] * ((X - X @ W) ** 2).sum()


def dataset_description(dataset):
    return "_".join([str(i) for i in list(dataset.description)])


def dataset_dirname(dataset):  # without random seed
    return "_".join([str(i) for i in list(dataset.description)[:-1]])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pk_files(path):
    return [pk.load(open(i, "rb")) for i in list(Path(path).rglob("*.pk"))]


def load_results(path):
    """ Load results df and recover np arrays for adjacency matrices. A little hacky. """
    def _eval_ndarray_string(x):
        return [eval("np."+x[1:-1])]
    results = pd.read_csv(path)
    results["scaling_factors"] = results["scaling_factors"].apply(_eval_ndarray_string)
    results["W_true"] = results["W_true"].apply(_eval_ndarray_string)
    results["W_est"] = results["W_est"].apply(_eval_ndarray_string)
    results["start_time"] = results["start_time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
    return results


def overwrite_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def create_folder(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass


def standardize(data):
    return data / data.std(0, keepdims=True)


def dagify(W):
    W_temp = np.where(W==0, np.inf, W)
    min_idx = np.unravel_index(np.argmin(np.abs(W_temp)), W.shape)
    W[min_idx] = 0.
    return W


def stop_pycausalvm():
    import javabridge
    javabridge.kill_vm()


def obtaincausalorder(Bin):
    B = Bin.copy()
    d = B.shape[0]
    idx = np.arange(d)
    pi = []
    for _ in range(d):
        succ = B.sum(0).argmin()
        pi.append(idx[succ])
        idx = np.delete(idx, succ, 0)
        B = np.delete(B, succ, 0)
        B = np.delete(B, succ, 1)
    return pi


def varsortability(X, W, tol=1e-9):
    """ Takes n x d data and a d x d adjaceny matrix,
    where the i,j-th entry corresponds to the edge weight for i->j,
    and returns a value indicating how well the variance order
    reflects the causal order. """
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)

    n_paths = 0
    n_correctly_ordered_paths = 0

    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
        n_correctly_ordered_paths += 1/2*(
            (Ek * var / var.T <= 1 + tol) *
            (Ek * var / var.T >  1 - tol)).sum()
        Ek = Ek.dot(E)

    return n_correctly_ordered_paths / n_paths


def shd_cpdag(B_true, B_est):
    """ 
    Both inputs are [d, d] matrices with entries {0, 1}
    where bidirected edges i <-> j are coded by placing a 1 at [i,j] and [j,i]
    """
    if ~np.isin(B_true, [0, 1]).any() or ~np.isin(B_est, [0, 1]).any():
        raise ValueError(
            'Both inputs should be CPDAG matrices with 0 and 1 only')

    d = B_true.shape[0]
    # code lower and upper triangle differently
    W = np.tril(np.ones((d, d)), -1) * 1. + np.triu(np.ones((d, d)), 1) * 2.
    # if we fold the weighted B matrices up, i.e. consider
    # W * B + (W * B).T
    # then 0, 1, 2, 3 code no edge/->/<-/<->
    WB_true = np.triu((W * B_true) + (W * B_true).T, 1)
    WB_est = np.triu((W * B_est) + (W * B_est).T, 1)

    # return count of mistakes, where mistaking any of
    # no edge/->/<-/<->
    # by one of the remaining three options counts as 1 mistake
    return (WB_true != WB_est).sum()


def sid(W, W_est):
    """ The first argument must be the ground-truth graph """
    res = r_sid.structIntervDist(W != 0, W_est != 0)
    return res.rx2('sid')[0]


def is_dag(W):
    """ Determine if graph formed by adj matrix W is DAG """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        raise ValueError('We do not want this utils function to act on CPDAGs')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            a = np.sqrt(6)/np.pi * scale
            z = np.random.gumbel(scale=a, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            a = scale * np.sqrt(3)
            z = np.random.uniform(low=-a, high=a, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
            gp-add-lach Lachapelle A.5
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    assert np.sum(B) == np.count_nonzero(B), 'Adjacency matrix can only contain 0s or 1s.'
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            if sem_type == 'gp-add-lach':
                # source nodes have [1,2] variance here
                z = np.random.normal(scale=np.sqrt(2.5)*scale, size=n)
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add' or sem_type == 'gp-add-lach':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if (noise_scale is not None) else np.ones(d)
    if sem_type == 'gp-add-lach':
        scale_vec = np.sqrt(np.random.uniform(low=.4, high=.8, size=d))
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X