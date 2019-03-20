"""
Balasubramanian Srinivasan and Ryan L Murphy
This code implements so-called RP-Duvenaud
(1) \harrow{f} is defined as the Graph Conv model based on Duvenaud and implemented by the deepchem team
(2) We assign unique one-hot identifiers to increase representational power, rendering the model perm-sensitive,
but wrapping it in our pooling makes it permutation invariant again
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import pickle
import tensorflow as tf
import deepchem as dc
from random import shuffle
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
tg = TensorGraph(use_queue=False)

TASK = sys.argv[1]  # 'tox_21', 'hiv', 'muv
METHOD = sys.argv[2]  # Either 'unique_ids' or 'unique_local'
RUN_NUM = sys.argv[3]

batch_size = 96
NUM_EPOCHS = 100
INFERENCE_TIME_PERMUTATIONS = 10


def generate_rp_vertex_feats_unique_ids(vertex_feats):
	"""
	Use the "unique-ids" scheme for assigning one-hot unique IDs to atoms.
	Unique scheme gives each and every atom in the molecule it's own ID, in contrast with the "local" method below

	Implicitly, max_atoms comes from the global env

	:param vertex_feats: Matrix of endowed vertex (atom) attributes Xv that come with the data
	:return: concat(Xv, Ids) where Ids is a matrix of one-hot identifiers for the atom
	"""
	K = vertex_feats.shape[0]
	appender = np.zeros((K, max_atoms))
	atom_permute = list(range(K))
	shuffle(atom_permute)
	for i in range(K):
		atom_number = atom_permute[0]
		atom_permute.pop(0)
		appender[i][atom_number] = 1
	vertex_feats_appended = np.concatenate((vertex_feats, appender),axis=1)
	return vertex_feats_appended


def generate_rp_vertex_feats_unique_local(vertex_feats):
	"""
	Use the "unique-local" scheme for assigning one-hot unique IDs to atoms.
	Here, atoms of the same type get unique one-hot IDS but atoms of a different type  might have the same ID

	For example, given two carbons and two hydrogens,
	(C, (1 ,0))
	(C, (0 ,1))
	(H, (1 ,0))
	(H, (0 ,1))

	Implicitly, max_atoms comes from the global env

	:param vertex_feats: Matrix of endowed vertex (atom) attributes Xv that come with the data
	:return: concat(Xv, Ids) where Ids is a matrix of one-hot identifiers for the atom
	"""
	K = vertex_feats.shape[0]
	appender = np.zeros((K, max_atoms))
	#
	# Find the number of atoms of each type
	# If there are m atoms, allocate a list [0, 1, ..., m-1] of identifiers
	#  that will get mapped to one-hot encodings
	#
	count_tracker = {}
	for i in range(K):
		atom = np.array_str(vertex_feats[i],max_line_width=300)
		if atom in count_tracker:
			count_tracker[atom].append(len(count_tracker[atom]))
		else :
			count_tracker[atom]=[0]
	#
	# Shuffle the lists: pi sgd
	#
	for item in count_tracker:
		shuffle(count_tracker[item])
	#
	# Map IDs to one-hot encodings
	#
	for i in range(K):
		atom = np.array_str(vertex_feats[i],max_line_width=300)
		unique_local_id = count_tracker[atom][0]
		count_tracker[atom].pop(0)
		appender[i][unique_local_id] = 1
	
	vertex_feats_appended = np.concatenate((vertex_feats, appender),axis=1)
	return vertex_feats_appended


def generate_new_X(dataset):
	new_array = []
	size = dataset.shape[0]
	for i in range(size):
		mol = dataset[i]
		atom_feats = mol.get_atom_features()
		if METHOD == 'unique_local':
			new_atom_feats = generate_rp_vertex_feats_unique_local(atom_feats)
		elif METHOD == 'unique_ids':
			new_atom_feats = generate_rp_vertex_feats_unique_ids(atom_feats)
		adjacent_list = mol.get_adjacency_list()
		new_mol = dc.feat.mol_graphs.ConvMol(new_atom_feats, adjacent_list)
		new_array.append(new_mol)
	return np.array(new_array)


def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
	for epoch in range(epochs):
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
	if TASK != 'hiv':
		n_samples = len(y_true)
		retval = np.stack(y_pred, axis=1)
		return retval[:n_samples]
	else :
		n_samples = len(y_true)
		retval = y_pred
		return retval[:n_samples]


def proper_inference():
	for iteration in range(INFERENCE_TIME_PERMUTATIONS):
		if TASK == 'tox_21':
			new_test_data = generate_new_X(train_dataset.X[5000:])
			rp_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, train_dataset.y[5000:], train_dataset.w[5000:] ,train_dataset.ids[5000:], data_dir=None)
		else:
			new_test_data = generate_new_X(test_dataset.X)
			rp_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, test_dataset.y, test_dataset.w ,test_dataset.ids, data_dir=None)
		preds = tg.predict_on_generator(data_generator(rp_dataset, predict=True))
		preds = reshape_y_pred(rp_dataset.y, preds)
		if iteration == 0:
			sum_out = np.zeros((preds.shape))
		sum_out += preds

	one_random_perm = preds
	sum_out = sum_out/float(INFERENCE_TIME_PERMUTATIONS)
	return (one_random_perm, sum_out)

#
# Set up logging information
#
default_stdout = sys.stdout
logger_file = str(TASK) + "_" + str(METHOD) + "_" + str(RUN_NUM) + ".log"
lfile = open(logger_file, 'w')
sys.stdout = lfile
val_roc_tracker = []
test_roc_tracker = []
val_pfile =  str(TASK) + "_" + str(METHOD) + "_" + str(RUN_NUM) + "_val.pickle"
test_pfile = str(TASK) + "_" + str(METHOD) + "_" + str(RUN_NUM) + "_test.pickle"

print("Dataset is ", TASK)
print("RP Technique used is ", METHOD)
print("Run number is ", RUN_NUM)
sys.stdout.flush()

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
current_tasks, current_datasets, transformers = dataloader(featurizer='GraphConv', reload=True, split='random')
train_dataset, valid_dataset, test_dataset = current_datasets

#
# Determine the max number of atoms
#
max_atoms = 0
for mol in train_dataset.X:
	num_atom = mol.get_num_atoms()
	if num_atom > max_atoms :
		max_atoms = num_atom
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Look ahead to largets molecule in test-valid
#
# We assume for our experiments that we can, in general, look ahead to the test set to find the
#  largest molecule.  This is needed to allocated a fixed-sized feature vector (padding with zeros as needed)
# Further discussion is found in our appendix.
#
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
if TASK!= 'tox_21':
	for mol in valid_dataset.X:
		num_atom = mol.get_num_atoms()
		if num_atom > max_atoms :
			max_atoms = num_atom
	for mol in test_dataset.X:
		num_atom = mol.get_num_atoms()
		if num_atom > max_atoms :
			max_atoms = num_atom
#
# Build up model object
#  Follow: https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html
#
atom_features = Feature(shape=(None, 75+max_atoms))
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

if TASK != 'tox_21':
	new_train_data = generate_new_X(train_dataset.X)
	new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y, train_dataset.w ,train_dataset.ids, data_dir=None)
	print("Train Data - added RP")
	new_valid_data = generate_new_X(valid_dataset.X)
	new_valid_dataset = dc.data.datasets.DiskDataset.from_numpy(new_valid_data, valid_dataset.y, valid_dataset.w ,valid_dataset.ids, data_dir=None)
	print("Valid Data - added RP")
	new_test_data = generate_new_X(test_dataset.X)
	new_test_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, test_dataset.y, test_dataset.w ,test_dataset.ids, data_dir=None)
	print("Test Data - added RP")
else :
	new_train_data = generate_new_X(train_dataset.X[:3800])
	new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y[:3800], train_dataset.w[:3800] ,train_dataset.ids[:3800], data_dir=None)
	print("Train Data - added RP - tox21")
	new_valid_data = generate_new_X(train_dataset.X[3800:5000])
	new_valid_dataset = dc.data.datasets.DiskDataset.from_numpy(new_valid_data, train_dataset.y[3800:5000], train_dataset.w[3800:5000] ,train_dataset.ids[3800:5000], data_dir=None)
	print("Valid Data - added RP - tox21")
	new_test_data = generate_new_X(train_dataset.X[5000:])
	new_test_dataset = dc.data.datasets.DiskDataset.from_numpy(new_test_data, train_dataset.y[5000:], train_dataset.w[5000:] ,train_dataset.ids[5000:], data_dir=None)
	print("Test Data - added RP - tox21")

metric = dc.metrics.Metric(
	dc.metrics.roc_auc_score, np.mean, mode="classification")

best_auc_score = 0.0
for i in range(NUM_EPOCHS):
	print("Epoch Num: ", i)
	sys.stdout.flush()
	tg.fit_generator(data_generator(new_train_dataset, epochs=1))
	if TASK != 'tox_21':
		new_train_data = generate_new_X(train_dataset.X)
		new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y, train_dataset.w ,train_dataset.ids, data_dir=None)
	else :
		new_train_data = generate_new_X(train_dataset.X[:3800])
		new_train_dataset = dc.data.datasets.DiskDataset.from_numpy(new_train_data, train_dataset.y[:3800], train_dataset.w[:3800] ,train_dataset.ids[:3800], data_dir=None)
	print("Validation Loss")
	valid_predictions = tg.predict_on_generator(data_generator(new_valid_dataset, predict=True))
	valid_predictions = reshape_y_pred(new_valid_dataset.y, valid_predictions)
	valid_scores = metric.compute_metric(new_valid_dataset.y, valid_predictions, new_valid_dataset.w)
	print("Valid ROC-AUC Score: %f" % valid_scores)
	val_roc_tracker.append(valid_scores)
	if valid_scores > best_auc_score:
		best_auc_score = valid_scores
		one_random_perm, sum_out = proper_inference()

		test_scores = metric.compute_metric(new_test_dataset.y, one_random_perm, new_test_dataset.w)
		print("test ROC-AUC 1 inference Score: %f" % test_scores)
		test_scores = metric.compute_metric(new_test_dataset.y, sum_out, new_test_dataset.w)
		print("test ROC-AUC proper inference Score: %f" % test_scores)
		test_roc_tracker.append(test_scores)
	with open(val_pfile, 'wb') as file:
		pickle.dump(val_roc_tracker, file)
	with open(test_pfile, 'wb') as file:
		pickle.dump(test_roc_tracker, file)


print("Evaluating model")
train_predictions = tg.predict_on_generator(data_generator(new_train_dataset, predict=True))
train_predictions = reshape_y_pred(new_train_dataset.y, train_predictions)
train_scores = metric.compute_metric(new_train_dataset.y, train_predictions, new_train_dataset.w)
print("Training ROC-AUC Score: %f" % train_scores)

valid_predictions = tg.predict_on_generator(data_generator(new_valid_dataset, predict=True))
valid_predictions = reshape_y_pred(new_valid_dataset.y, valid_predictions)
valid_scores = metric.compute_metric(new_valid_dataset.y, valid_predictions, new_valid_dataset.w)
print("Valid ROC-AUC Score: %f" % valid_scores)


one_random_perm, sum_out = proper_inference()
test_predictions = tg.predict_on_generator(data_generator(new_test_dataset, predict=True))
test_predictions = reshape_y_pred(new_test_dataset.y, test_predictions)
test_scores = metric.compute_metric(new_test_dataset.y, one_random_perm, new_test_dataset.w)
print("test ROC-AUC 1 inference Score: %f" % test_scores)
test_scores = metric.compute_metric(new_test_dataset.y, sum_out, new_test_dataset.w)
print("test ROC-AUC 20 inference Score: %f" % test_scores)


sys.stdout = default_stdout
lfile.close()
