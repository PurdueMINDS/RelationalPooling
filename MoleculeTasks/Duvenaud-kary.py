"""
Balasubramanian Srinivasan and Ryan L Murphy
This code implements so-called k-ary RP approaches
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
import sys

from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
from random import shuffle
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
tg = TensorGraph(use_queue=False)


TASK = sys.argv[1]  # 'tox_21', 'hiv', 'muv
K = int(sys.argv[2])
technique = 'dfs'
batch_size = 96
NUM_EPOCHS = 100


def randomize_perm(a):
    ordering = list(range(a))
    shuffle(ordering)
    return ordering


def depth_first_search(neighbour_list, root_node):
    """ DFS can be used as a poly-canonical ordering to reduce computational cost in the RP sum """
    visited_nodes = set()
    order = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node not in visited_nodes:
            visited_nodes.add(node)
            order.append(node)
            stack.extend(set(neighbour_list[node]) - visited_nodes)
    return order


def breadth_first_search(neighbour_list, root_node):
    """ BFS can be used as a poly-canonical ordering to reduce computational cost in the RP sum """
    visited_nodes = set()
    order = []
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        if node not in visited_nodes:
            visited_nodes.add(node)
            order.append(node)
            queue.extend(set(neighbour_list[node]) - visited_nodes)
    return order


def generate_new_X(dataset, K, technique):
    """ Reduce to k-ary and run poly-canonical ordering"""
    count = 0
    new_array = []
    size = dataset.shape[0]
    for i in range(size):
        mol = dataset[i]
        min_degree, max_degree = 1000, 0
        atom_feats = mol.get_atom_features()
        adjacent_list = mol.get_adjacency_list()
        num_atoms = mol.get_num_atoms()
        if num_atoms > K:
            #Reduce to k-ary
            count+=1
            ordering = randomize_perm(num_atoms)
            if technique == 'dfs':
                order = depth_first_search(adjacent_list,ordering[0])
            elif technique == 'bfs':
                order = breadth_first_search(adjacent_list,ordering[0])
            else :
                order = ordering
            if (len(order) < K):
                  order = ordering
            order = order[:K]
            atom_feats = atom_feats[order]
            new_atom_feats = atom_feats
            create_adjacency = []
            for i in order:
                edges = []
                for neighbor in adjacent_list[i]:
                    if neighbor in order:
                        get_new_index = int(order.index(neighbor))
                        edges.append(get_new_index)
                create_adjacency.append(edges)
            new_mol = dc.feat.mol_graphs.ConvMol(new_atom_feats, create_adjacency)
        else :
            new_mol = dc.feat.mol_graphs.ConvMol(atom_feats, adjacent_list)
        new_array.append(new_mol)
    print(count)
    return np.array(new_array)


def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
    for epoch in range(epochs):
        if not predict:
            print('Starting epoch %i' % epoch)
        for ind, (X_b, y_b, w_b, ids_b) in enumerate(
            dataset.iterbatches(batch_size, pad_batches=pad_batches, deterministic=True)):
            d = {}
            for index, label in enumerate(labels):
                d[label] = to_one_hot(y_b[:, index])
            d[weights] = w_b
            multiConvMol = ConvMol.agglomerate_mols(X_b)
            d[atom_features] = multiConvMol.get_atom_features()
            d[degree_slice] = multiConvMol.deg_slice
            d[membership] = multiConvMol.membership
            for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
            yield d


def reshape_y_pred(y_true, y_pred):
    """
    TensorGraph.Predict returns a list of arrays, one for each output
    We also have to remove the padding on the last batch
    Metrics taks results of shape (samples, n_task, prob_of_class)
    """
    n_samples = len(y_true)
    retval = np.stack(y_pred, axis=1)
    return retval[:n_samples]


if TASK == 'tox_21':
    from deepchem.molnet import load_tox21 as dataloader
    NUM_TASKS = 12
elif TASK == 'hiv':
    from deepchem.molnet import load_hiv as dataloader
    NUM_TASKS = 1
elif TASK == 'muv':
    from deepchem.molnet import load_muv as dataloader
    NUM_TASKS = 17

# -------------------------------------------------
#  Load datasets, tasks, and transformers
#  The number of tasks in each dataset can be found in Table 1 of MoleculeNet: A Benchmark for Molecular Machine Learning
#   by Wu et. al.
current_tasks, current_datasets, transformers = dataloader(featurizer='GraphConv',reload=True,split='random')
train_dataset, valid_dataset, test_dataset = current_datasets
#
# Build up model object
#  Follow: https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html
#
atom_features = Feature(shape=(None, 75))
degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
membership = Feature(shape=(None,), dtype=tf.int32)

deg_adjs = []
for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)


gc1 = GraphConv(
    64,
    activation_fn=tf.nn.relu,
    in_layers=[atom_features, degree_slice, membership] + deg_adjs)
batch_norm1 = BatchNorm(in_layers=[gc1])
gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)
gc2 = GraphConv(
    64,
    activation_fn=tf.nn.relu,
    in_layers=[gp1, degree_slice, membership] + deg_adjs)
batch_norm2 = BatchNorm(in_layers=[gc2])
gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)
dense = Dense(out_channels=128, activation_fn=tf.nn.relu, in_layers=[gp2])
batch_norm3 = BatchNorm(in_layers=[dense])
readout = GraphGather(
    batch_size=batch_size,
    activation_fn=tf.nn.tanh,
    in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)

costs = []
labels = []
for task in range(len(current_tasks)):
    classification = Dense(
        out_channels=2, activation_fn=None, in_layers=[readout])

    softmax = SoftMax(in_layers=[classification])
    tg.add_output(softmax)

    label = Label(shape=(None, 2))
    labels.append(label)
    cost = SoftMaxCrossEntropy(in_layers=[label, classification])
    costs.append(cost)

all_cost = Stack(in_layers=costs, axis=1)
weights = Weights(shape=(None, len(current_tasks)))
loss = WeightedError(in_layers=[all_cost, weights])
tg.set_loss(loss)
# Data splits
# Tox21 is treated differently: we manually (randomly) split into test, train, and valid directly from train_dataset.X
#   (rather than letting deepchem provide the data directly)
# Reason: In the early stages of developing the code, the valid_dataset and test_dataset were empty for tox and
#         we observed a comment in the deepchem source code leading us to believe this was intended.
#         Thus, when we access valid_dataset.X and test_dataset.X, we don't do it for tox21. We only later
#         found that we could access tox21 validation and test. But we do this for all models, so the treatment is fair
#
#
# This treatment is done for all models, so the comparison is fair.
#
if TASK != 'tox_21':
    new_train_data = generate_new_X(train_dataset.X, K, technique)
    new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y, train_dataset.w ,train_dataset.ids, data_dir=None)
    print("Train Data - added RP")
    new_valid_data = generate_new_X(valid_dataset.X, K, technique)
    new_valid_dataset = dc.data.datasets.DiskDataset.from_numpy(new_valid_data, valid_dataset.y, valid_dataset.w ,valid_dataset.ids, data_dir=None)
    print("Valid Data - added RP")
    new_test_data = generate_new_X(test_dataset.X, K, technique)
    new_test_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, test_dataset.y, test_dataset.w ,test_dataset.ids, data_dir=None)
    print("Test Data - added RP")
else:
    new_train_data = generate_new_X(train_dataset.X[:3800], K, technique)
    new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y[:3800], train_dataset.w[:3800] ,train_dataset.ids[:3800], data_dir=None)
    print("Train Data - added RP - tox21")
    new_valid_data = generate_new_X(train_dataset.X[3800:5000], K, technique)
    new_valid_dataset = dc.data.datasets.DiskDataset.from_numpy(new_valid_data, train_dataset.y[3800:5000], train_dataset.w[3800:5000] ,train_dataset.ids[3800:5000], data_dir=None)
    print("Valid Data - added RP - tox21")
    new_test_data = generate_new_X(train_dataset.X[5000:], K, technique)
    new_test_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, train_dataset.y[5000:], train_dataset.w[5000:] ,train_dataset.ids[5000:], data_dir=None)
    print("Test Data - added RP - tox21")


tg.fit_generator(data_generator(new_train_dataset, epochs=NUM_EPOCHS))

metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")


print("Evaluating model")
train_predictions = tg.predict_on_generator(data_generator(new_train_dataset, predict=True))
train_predictions = reshape_y_pred(new_train_dataset.y, train_predictions)
train_scores = metric.compute_metric(new_train_dataset.y, train_predictions, new_train_dataset.w)
print("Training ROC-AUC Score: %f" % train_scores)

valid_predictions = tg.predict_on_generator(data_generator(new_valid_dataset, predict=True))
valid_predictions = reshape_y_pred(new_valid_dataset.y, valid_predictions)
valid_scores = metric.compute_metric(new_valid_dataset.y, valid_predictions, new_valid_dataset.w)
print("Valid ROC-AUC Score: %f" % valid_scores)

test_predictions = tg.predict_on_generator(data_generator(new_test_dataset, predict=True))
test_predictions = reshape_y_pred(new_test_dataset.y, test_predictions)
test_scores = metric.compute_metric(new_test_dataset.y, test_predictions, new_test_dataset.w)
print("test ROC-AUC Score: %f" % test_scores)


