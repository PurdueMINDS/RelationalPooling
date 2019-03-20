"""
Balasubramanian Srinivasan and Ryan L Murphy
This code implements neural-network-based methods as \harrow{f} in the RP framework
"""

import torch
import torch.nn as nn
import numpy as np
import time
import pickle
import sys
from itertools import permutations, product
from scipy import sparse
from torch.nn import init
from random import shuffle
from sklearn.metrics import roc_auc_score

TASK = sys.argv[1]
batch_size = 96
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def align_adjacency(a, b):
    data0 = a.get_atom_features()
    data1 = b.get_atom_features()
    index = list(range(data0.shape[0]))
    remain = list(range(data1.shape[0]))
    mapping_dict = dict((k, k) for k in index)
    for i in index:
        for j in remain:
            if np.array_equal(data0[i], data1[j]):
                mapping_dict[i] = j
                remain.remove(j)
                break
    return mapping_dict


def randomize_perm(a):
    ordering = list(range(a))
    shuffle(ordering)
    return ordering


def permute_array(a, ordering, mapping_dict):
    pair_features = a.get_pair_features()
    new_array = np.zeros(pair_features.shape)
    mod_factor = pair_features.shape[0]
    m, n = 0, 0
    for i in ordering:
        for j in ordering:
            new_array[m][n] = pair_features[mapping_dict[i]][mapping_dict[j]]
            n += 1
            n = n % mod_factor
        m += 1
        m = m % mod_factor
    return new_array


def depth_first_search(neighbour_list, root_node):
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


def construct_tensor(dataset_conv, dataset_weave, y):
    size = dataset_weave.shape[0]
    arr_pair = []
    arr_indv = []
    y_true = []
    for i in range(size):
        a = dataset_conv[i]
        b = dataset_weave[i]
        ordering = randomize_perm(a.get_num_atoms())
        mapping_dict = align_adjacency(a, b)
        order = depth_first_search(a.get_adjacency_list(), ordering[0])
        pair_array = permute_array(b, order, mapping_dict)
        indv_array = a.get_atom_features()[order]
        if len(pair_array) == len(indv_array):
            arr_pair.append(torch.Tensor(pair_array).to(device))
            arr_indv.append(torch.Tensor(indv_array).to(device))
            y_true.append(y[i])
    return (arr_indv, arr_pair, y_true)


def unison_shuffled(a, b, c):
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn_unit_1 = nn.LSTM(14, 100, batch_first=True)
        self.indv_linear_1 = nn.Linear(75, 100)
        init.xavier_uniform_(self.indv_linear_1.weight)
        self.indv_act_1 = nn.ReLU()
        self.rnn_unit_2 = nn.LSTM(200, 100, batch_first=True)
        self.rho_lin_1 = nn.Linear(100, 100)
        init.xavier_uniform_(self.rho_lin_1.weight)
        self.rho_act_1 = nn.ReLU()
        self.final_lin = nn.Linear(100, NUM_TASKS)
        init.xavier_uniform_(self.final_lin.weight)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pair_inp, indv_inp):
        rho_input = torch.zeros((1, 100)).to(device)
        for i in range(len(pair_inp)):
            out_rnn_1, (h_n, c_n) = self.rnn_unit_1(pair_inp[i])
            out_indv = self.indv_linear_1(indv_inp[i])
            out_indv = self.indv_act_1(out_indv).unsqueeze(0)
            inp_rnn_2 = torch.cat((c_n, out_indv), 2)
            out_rnn_2, (h_n, c_n) = self.rnn_unit_2(inp_rnn_2)
            c_n = c_n.squeeze(0)
            rho_input = torch.cat((rho_input, c_n), 0)
        rho_input = rho_input[1:]
        rho_out = self.rho_lin_1(rho_input)
        rho_out = self.rho_act_1(rho_out)
        final_out = self.final_lin(rho_out)
        return final_out

    def compute_loss(self, pair_inp, indv_inp, y_true):
        pred = self.forward(pair_inp, indv_inp)
        return self.loss_func(pred, y_true)

    def compute_proba(self, pair_inp, indv_inp):
        return torch.sigmoid(self.forward(pair_inp, indv_inp))


if TASK == 'tox_21':
    from deepchem.molnet import load_tox21 as dataloader
    NUM_TASKS = 12
elif TASK == 'hiv':
    from deepchem.molnet import load_hiv as dataloader
    NUM_TASKS = 1
elif TASK == 'muv':
    from deepchem.molnet import load_muv as dataloader
    NUM_TASKS = 17


current_tasks_weave, current_datasets_weave, transformers_weave = dataloader(featurizer='Weave')
current_tasks_conv, current_datasets_conv, transformers_conv = dataloader(featurizer='GraphConv')

train_dataset_weave, valid_dataset_weave, test_dataset_weave = current_datasets_weave
train_dataset_conv, valid_dataset_conv, test_dataset_conv = current_datasets_conv

train_shuffled_conv = train_dataset_conv.X
train_shuffled_weave = train_dataset_weave.X
train_shuffled_y = train_dataset_conv.y

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
    train_pair = train_dataset_weave.X
    train_indv = train_dataset_conv.X
    train_y = train_dataset_conv.y
    valid_pair = valid_dataset_weave.X
    valid_indv = valid_dataset_conv.X
    valid_y = valid_dataset_conv.y
    test_pair = test_dataset_weave.X
    test_indv = test_dataset_conv.X
    test_y = test_dataset_conv.y 
else :
    train_shuffled_conv, train_shuffled_weave, train_shuffled_y = unison_shuffled(train_shuffled_conv, train_shuffled_weave, train_shuffled_y)
    train_indv = train_shuffled_conv[:3800]
    train_pair = train_shuffled_weave[:3800]
    train_y = train_shuffled_y[:3800]
    valid_indv = train_shuffled_conv[3800:5000]
    valid_pair = train_shuffled_weave[3800:5000]
    valid_y = train_shuffled_y[3800:5000]
    test_indv = train_shuffled_conv[5000:]
    test_pair = train_shuffled_weave[5000:]
    test_y = train_shuffled_y[5000:]

# train_shuffled_conv, train_shuffled_weave, train_shuffled_y = unison_shuffled(train_shuffled_conv, train_shuffled_weave, train_shuffled_y)

#Construct Valid and Test 
indv, pair, y_true = construct_tensor(train_indv,train_pair, train_y)
valid_indv, valid_pair, valid_y_true = construct_tensor(valid_indv, valid_pair, valid_y)
test_indv, test_pair, test_y_true = construct_tensor(test_indv, test_pair, test_y)

# Train over multiple epochs
val_score_tracker, train_loss_tracker = [], []
NUM_TRAINING_EXAMPLES = len(indv)
start_time = time.time()
num_batches = int(NUM_TRAINING_EXAMPLES / batch_size)
val_loss_tracker = []
num_steps_tracker = []
checkpoint_file_name = "rnn_dfs_{}.model".format(TASK)
checker = RNNModel().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, checker.parameters()), lr=0.003)
count = 0
best_roc_auc = 0.0

for epoch in range(num_epochs):
    print("Epoch Num: ", epoch)
    # Do seed and random shuffle of the input
    print("Performing Random Shuffle")
    train_indv, train_pair, train_y = unison_shuffled(train_indv,train_pair, train_y)
    indv, pair, y_true = construct_tensor(train_indv,train_pair, train_y)
    y_true_tensor = torch.FloatTensor(y_true).to(device)
    print("Random Shuffle Done")
    for batch in range(num_batches):
        optimizer.zero_grad()
        batch_pair = pair[batch_size * batch:batch_size * batch + batch_size]
        batch_indv = indv[batch_size * batch:batch_size * batch + batch_size]
        batch_y = y_true_tensor[batch_size * batch:batch_size * batch + batch_size]
        loss = checker.compute_loss(batch_pair, batch_indv, batch_y)
        loss.backward()
        optimizer.step()
        count += 1
        
        if count % 100 == 0:
            with torch.no_grad():
                val_loss = checker.compute_loss(valid_pair, valid_indv, torch.FloatTensor(valid_y_true).to(device))
                val_loss_tracker.append(val_loss.item())
                pickle.dump(val_loss_tracker, open("val_loss_dfs_rnn_{}.p".format(TASK), "wb"))
                print("Val Loss at Step ", count, " : ", val_loss.item())
                num_steps_tracker.append(count)

                val_out = checker.compute_proba(valid_pair, valid_indv)
                val_y_pred = np.round(val_out.detach().cpu().numpy())
                val_score = roc_auc_score(np.array(valid_y_true), val_y_pred)
                val_score_tracker.append(val_score)
                pickle.dump(val_score_tracker, open("val_score_dfs_rnn_{}.p".format(TASK), "wb" ))
                if val_score > best_roc_auc:
                    print("Best Val ROC AUC Score till now: ", val_score)
                    best_roc_auc = val_score
                    torch.save(checker.state_dict(),checkpoint_file_name)

    with torch.no_grad():
        loss = checker.compute_loss(pair, indv, y_true_tensor)
        print("Epoch Training Loss: ", loss.item())
        train_loss_tracker.append(loss.item())
        pickle.dump(train_loss_tracker, open("train_loss_dfs_rnn_{}.p".format(TASK), "wb" ))
        

end_time = time.time()
total_training_time = end_time - start_time
print("Total Time: ", total_training_time)

#
# Run test-set prediction (TODO: paste separate script that uses trained model here)
#

