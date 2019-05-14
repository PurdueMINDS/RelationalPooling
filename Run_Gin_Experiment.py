"""
(anon). Synthetic experiments with GIN and RP-GIN

Usage:
    Run_Gin_Experiment.py (--cv-fold <N>) (--out-weight-dir <folder>) (--out-log-dir <folder>) [--use-batchnorm] [--dense-dropout-prob <float>]
                          [--num-mlp-hidden <N>] [--num-gnn-layers <N>]
                          [--model-type <string>]
                          [--set-epsilon-zero] [--vertex-embed-dim <N>]
                          [--mlp-hidden-dim <N>] [--learning-rate <float>]
                          [--num-epochs <N>] [--num-inf-perm <N>]
                          [--onehot-id-dim <N>] [--seed-val <N>]

Options:
    --cv-fold <N>                   Which fold in cross-validation: 0 thru 5
    --out-weight-dir <folder>       Output directory where trained weights (and any other objects) will be stored
    --out-log-dir <folder>          Output directory where logfiles will be saved
    --use-batchnorm                 Boolean flag, should batch normalization be implemented?
    --dense-dropout-prob <float>    Dropout probability for the dense layer [default: 0.0]
    --num-mlp-hidden <N>            Number of hidden layers in the MLP [default: 2]
    --num-gnn-layers <N>            Number of iterations of WL-like aggregation [default: 5]
    --model-type <string>           Either 'regularGin' or 'dataAugGin' or 'rpGin. Note: the model choice influences how the data is loaded/used [default: regularGin]
    --set-epsilon-zero              Boolean flag, should epsilon be set to zero?  By default, we train epsilon via backprop
    --vertex-embed-dim <N>          Dimension of each vertex's embedding [default: 16]
    --mlp-hidden-dim <N>            Number of hidden units in the aggregator's multilayer perceptron [default: 16]
    --learning-rate <float>         Learning rate for Adam Optimizer [default: 0.01]
    --num-epochs <N>                Number of epochs for training [default: 200]
    --num-inf-perm <N>              Number of inference-time permutations [default: 5]
    --onehot-id-dim <N>             For use with rpGin.  Dimension of the one-hot ID. [default: 41]
    --seed-val <N>                  Seed value, to get different random inits and variability [default: 1337]
"""
# python Run_Gin_Experiment.py --cv-fold 0 --model-type regularGin --num-epochs 100 --out-weight-dir some/folder --out-log-dir some/other/folder
#
import docopt
import os
import pickle
import random
from GIN.GIN_model import *
from GIN.GIN_utils import construct_onehot_ids
from training_utils import *
from schema import Schema, Use, And, Or
from operator import itemgetter

def get_filename_prefix(args):
    """ Create a string to name weights file, log file, etc."""
    prefix = args['--model-type'] + "_cv_" + str(args['--cv-fold'])
    if args['--use-batchnorm']:
        prefix += "batchnorm"

    if args['--dense-dropout-prob'] > 0.0:
        prefix += "_dropout_{}".format(args['--dense-dropout-prob'])

    if args['--set-epsilon-zero']:
        prefix += "_no_epsilon"

    if args['--num-gnn-layers'] != 5:
        prefix += "_gnn_layers_{}".format(args['--num-gnn-layers'])

    if args['--num-mlp-hidden'] != 2:
        prefix += "_mlp_hidden_{}".format(args['--num-mlp-hidden'])

    if args['--num-inf-perm'] != 5:
        prefix += "_num_inf_perm_{}".format(args['--num-inf-perm'])

    if args['--onehot-id-dim'] != 41:
        prefix += "_onehot_id_dim_{}".format(args['--onehot-id-dim'])

    prefix += "_s" + str(args['--seed-val']) + "_epochs_" + str(args['--num-epochs']) + "_"
    return prefix


def get_train_val_idx(num_graphs, cv_fold):
    """ Return a tuple of the train and val indices,
    depending on the cv_fold
    This method shuffles the index (with a seed)
    The shuffle is consistent across machines with python3"""
    #
    # Extract indices of train and val in terms of the shuffled list
    # Balanced across test and train
    # Assumes 10-class
    #
    random.seed(1)
    num_classes = 10
    num_per_class = int(num_graphs/num_classes)
    idx_to_classes = {}
    val_idx = []
    train_idx = []
    for cc in range(num_classes):
        idx_to_classes[cc] = list(range(cc*num_per_class, (cc+1)*num_per_class))
        random.shuffle(idx_to_classes[cc])
        # These indices correspond to the validation for this class.
        class_val_idx = slice(cv_fold * 3, cv_fold * 3 + 3, 1)
        # Extract validation.
        vals = idx_to_classes[cc][class_val_idx]
        val_idx.extend(vals)
        train_idx.extend(list(set(idx_to_classes[cc]) - set(vals)))
    #
    return tuple(train_idx), tuple(val_idx)

def accuracy(yhat, y, print_scores=False):
    """ Compute accuracy """
    scores = torch.argmax(yhat, dim=1)
    if print_scores:
        logging.info(scores)
    num_correct = torch.sum(scores == y).item()
    return num_correct/float(len(y))

if __name__ == '__main__':
    requirements = {
        '--use-batchnorm': Use(bool),
        '--dense-dropout-prob': And(Use(float), lambda fff: 0.0 <= fff < 1.0),
        '--num-mlp-hidden': Use(int),
        '--num-gnn-layers': Use(int),
        '--cv-fold': And(Use(int), lambda nnn: 0 <= nnn < 5),
        '--out-weight-dir': Use(str),
        '--out-log-dir': Use(str),
        '--model-type': And(Use(str), lambda sss: sss in ['regularGin', 'dataAugGin', 'rpGin']),
        '--set-epsilon-zero': Use(bool),
        '--vertex-embed-dim': And(Use(int), lambda mmm: mmm > 0),
        '--mlp-hidden-dim': And(Use(int), lambda lll: lll > 0),
        '--learning-rate': And(Use(float), lambda flo: flo > 0.0),
        '--num-epochs': And(Use(int), lambda epo: epo > 9),
        '--num-inf-perm': Use(int),
        '--onehot-id-dim': And(Use(int), lambda idd: idd > 0),
        '--seed-val': Use(int)
    }
    args = docopt.docopt(__doc__)
    args = Schema(requirements).validate(args)
    assert os.path.isdir(args['--out-weight-dir']), "Must enter a valid output weights directory"
    assert os.path.isdir(args['--out-log-dir']), "Must enter a valid output logs directory"
    #
    # Set up paths for logging and weight saving.
    #
    base_dir = os.getcwd()
    filename_pre = get_filename_prefix(args)
    log_file = os.path.join(args['--out-log-dir'],
                            filename_pre + '.log')

    weights_file = os.path.join(args['--out-weight-dir'],
                                filename_pre + '.pth')
    training_metrics_file = os.path.join(args['--out-weight-dir'],
                                         filename_pre + '.pkl')

    set_logger(log_file)
    logging.info(args)
    #
    # Load graphs, y
    #
    logging.info("---Loading data...---")
    sparse_adjmats = pickle.load(open(os.path.join(base_dir, 'Synthetic_Data', 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
    y = torch.load(os.path.join(base_dir, 'Synthetic_Data', 'y_Kary_Deterministic_Graphs.pt'))

    num_graphs = len(sparse_adjmats)
    logging.info("{} Adjacency matrices were loaded".format(num_graphs))
    #
    # Load X
    # Standard WL-approach: featureless implies use a constant vertex attribute, for every vertex
    # (such data could be generated here rather than loaded, but this coding structure easily
    # lends itself to future extensions)
    #
    if args['--model-type'] == 'regularGin':
        # X_all = torch.load(os.path.join(base_dir, 'Synthetic_Data', 'X_unity_Kary_Deterministic_Graphs.pt'))
        X_list = pickle.load(open(os.path.join(base_dir, 'Synthetic_Data', 'X_unity_list_Kary_Deterministic_Graphs.pkl'), 'rb'))
    elif args['--model-type'] == 'dataAugGin':
        X_list = pickle.load(open(os.path.join(base_dir, 'Synthetic_Data', 'X_eye_list_Kary_Deterministic_Graphs.pkl'), 'rb'))
    elif args['--model-type'] == 'rpGin':
        #
        # Set the dimension of the one hot id
        #   (redefine it if the user makes it too big)
        largest_adjmat = np.max([adjmat.shape[0] for adjmat in sparse_adjmats])
        if args['--onehot-id-dim'] > largest_adjmat:
            logging.info("Your selected value of onehot-id-dim, {}, is larger than the largest graph".format(args['--onehot-id-dim']))
            logging.info("I am resetting onehot-id-dim = {}, the largest adjmat".format(largest_adjmat))
            onehot_id_dim = largest_adjmat
        else:
            onehot_id_dim = args['--onehot-id-dim']
        #
        # Construct one hot ids
        #
        X_list = []
        for mat in sparse_adjmats:
            X_list.append(construct_onehot_ids(mat.shape[0], onehot_id_dim))
    #
    #  split according to cv fold
    #
    logging.info("---splitting into training and validation folds---")
    logging.info(" The indices are shuffled, and the shuffle is consistent on many machines as long as python3 is used")
    train_idx, val_idx = get_train_val_idx(num_graphs, args['--cv-fold'])

    train_adjmats = list(itemgetter(*train_idx)(sparse_adjmats))
    val_adjmats = list(itemgetter(*val_idx)(sparse_adjmats))
    y_train = y[torch.tensor(train_idx)]
    y_val = y[torch.tensor(val_idx)]
    #
    # Print class distribution
    #
    logging.info("------Class distributions---------")
    logging.info("train:")
    logging.info(np.unique(y_train.numpy(), return_counts=True))
    logging.info("test:")
    logging.info(np.unique(y_val.numpy(), return_counts=True))

    X_train = torch.cat(itemgetter(*train_idx)(X_list), dim=0)
    X_val = torch.cat(itemgetter(*val_idx)(X_list), dim=0)
    #
    # Define model
    #
    torch.manual_seed(args['--seed-val'])
    np.random.seed(args['--seed-val'])  # Used with rpGin, since random permutations are generated with scipy sparse (which uses np seed)
    logging.info("Building model...")

    if args['--use-batchnorm']:
        other_mlp_params = {'batchnorm': True}
    else:
        other_mlp_params = {}

    if args['--set-epsilon-zero']:
        eps_tunable = False
    else:
        eps_tunable = True

    if args['--model-type'] == 'regularGin':
        model = GinMultiGraph(adjmat_list=train_adjmats,
                              input_data_dim=X_train.shape[1],
                              num_agg_steps=args['--num-gnn-layers'],
                              vertex_embed_dim=args['--vertex-embed-dim'],
                              mlp_num_hidden=args['--num-mlp-hidden'],
                              mlp_hidden_dim=args['--mlp-hidden-dim'],
                              vertices_are_onehot=False,
                              target_dim=10,
                              epsilon_tunable=eps_tunable,
                              dense_layer_dropout=args['--dense-dropout-prob'],
                              other_mlp_parameters=other_mlp_params)

    elif args['--model-type'] == 'dataAugGin':
        model = GinMultiGraph(adjmat_list=train_adjmats,
                              input_data_dim=X_train.shape[1],
                              num_agg_steps=args['--num-gnn-layers'],
                              vertex_embed_dim=args['--vertex-embed-dim'],
                              mlp_num_hidden=args['--num-mlp-hidden'],
                              mlp_hidden_dim=args['--mlp-hidden-dim'],
                              vertices_are_onehot=True,
                              target_dim=10,
                              epsilon_tunable=eps_tunable,
                              dense_layer_dropout=args['--dense-dropout-prob'],
                              other_mlp_parameters=other_mlp_params)

    elif args['--model-type'] == 'rpGin':
        model = RpGin(adjmat_list=train_adjmats,
                      input_data_dim=X_train.shape[1],
                      num_agg_steps=args['--num-gnn-layers'],
                      vertex_embed_dim=args['--vertex-embed-dim'],
                      mlp_num_hidden=args['--num-mlp-hidden'],
                      mlp_hidden_dim=args['--mlp-hidden-dim'],
                      target_dim=10,
                      featureless_case=True,
                      vertices_are_onehot=True,
                      epsilon_tunable=eps_tunable,
                      dense_layer_dropout=args['--dense-dropout-prob'],
                      other_mlp_parameters=other_mlp_params)

    logging.info(model)
    #
    # Train
    #
    metrics = {'acc_train': [], 'acc_val': [], 'loss_train': [], 'loss_val': []}

    logging.info("------Training Model---------")
    learning_rate = args['--learning-rate']
    num_epochs = args['--num-epochs']

    logging.info("Train X has shape {}".format(X_train.shape))
    logging.info("Val X has shape {}".format(X_val.shape))
    logging.info("Train y has shape {}".format(y_train.shape))
    logging.info("Validation y has shape {}".format(y_val.shape))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_adjmats, X_train)
        loss_train = loss_func(pred, y_train)
        loss_train.backward()
        optimizer.step()
        #
        # Evaluate model.
        #  > loss and accuracy over validation
        #  > accuracy over train
        model.eval()
        with torch.no_grad():
            pred_val = model(val_adjmats, X_val)
            loss_val = loss_func(pred_val, y_val)

            # get accuracy and print predictions if it's regular GIN
            acc_train = accuracy(pred, y_train, print_scores=(epoch % 10 == 0))
            acc_val = accuracy(pred_val, y_val, print_scores=(epoch % 10 == 0))

            logging.info("~"*5)
            logging.info(
                "Epoch: %3d | Train Loss: %.5f | Val Loss: %.5f | Train Accuracy : %.5f | Val Accuracy : %.5f" % (epoch, loss_train, loss_val, acc_train, acc_val))

            metrics['acc_train'].append(acc_train)
            metrics['acc_val'].append(acc_val)
            metrics['loss_val'].append(loss_val.item())
            metrics['loss_train'].append(loss_train.item())

    if args['--model-type'] == 'rpGin':
        with torch.no_grad():
            pred_inf = model.inference(val_adjmats, X_val, args['--num-inf-perm'])
            final_accuracy = accuracy(pred_inf, y_val)
            logging.info("="*10)
            logging.info("Final accuracy: {}".format(final_accuracy))
            logging.info("="*10)
            metrics['final_accuracy'] = final_accuracy
    #
    # Save model
    #
    logging.info("Saving model to file")
    logging.info(weights_file)
    torch.save(model.state_dict(), weights_file)
    logging.info("...done saving")
    #
    # Save metrics
    #
    logging.info("Saving metrics")
    logging.info(training_metrics_file)
    pickle.dump(metrics, open(training_metrics_file, 'wb'))
    logging.info("... done saving")
