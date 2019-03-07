##########################################
#
# (anon), 2019
#
# Implement Graph Isomorphism Network (GIN) https://arxiv.org/pdf/1810.00826.pdf
# for the purposes of our project
#
##########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import scipy.sparse as sps
from training_utils import *
from torch.nn import init
from sklearn.utils import shuffle as sparse_shuffle

class MLP(nn.Module):
    """Define a multilayer perceptron
       assume that all intermediate hidden layers have the same dimension (number of neurons)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers, act=F.relu, other_mlp_parameters={}):
        """ :param: other_mlp_parameters: dictionary with keys of dropout and/or batchnorm.  values are dropout prob"""
        super(MLP, self).__init__()
        assert num_hidden_layers > 0, "MLP should have at least one hidden layer"
        assert isinstance(other_mlp_parameters, dict), 'other_mlp_parameters should be dict or none.'

        # Check that the other mlp parameters are valid
        for key_ in other_mlp_parameters.keys():
            if key_ not in ['dropout', 'batchnorm']:
                raise ValueError("The key entered into other_mlp_parameters is invalid.  Must be in ['dropout', 'batchnorm'].  Entered: "+ str(key_))

        if 'dropout' in other_mlp_parameters:
            assert isinstance(other_mlp_parameters['dropout'], float), "dropout prob should be a float"
            assert 0.0 <= other_mlp_parameters['dropout'] < 1.0, "dropout prob needs to be in half-open interval [0, 1)"
            self.dropout_prob = other_mlp_parameters['dropout']
            self.do_dropout = True
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)
            logging.info("Dropout will be used in MLP.  Probability: {}".format(self.dropout_prob))
        else:
            self.do_dropout = False
            logging.info("Dropout will NOT be used in MLP.")

        # Set batchnorm flags, add layers later
        # If batchnorm is a key and its value is true:
        if 'batchnorm' in other_mlp_parameters:
            assert isinstance(other_mlp_parameters['batchnorm'], bool), "batchnorm value must be bool"
            if other_mlp_parameters['batchnorm']:
                self.do_batchnorm = True
                logging.info("batchnorm WILL be applied in MLP")
        # If (batchnorm is not a key) OR (it has value False)
        else:
            self.do_batchnorm = False
            logging.info("batchnorm will NOT be applied in MLP.")

        if self.do_dropout and self.do_batchnorm:
            raise Warning("User selected both batchnorm and dropout in the MLP")

        self.act = act
        self.num_hidden_layers = num_hidden_layers
        self.layers = []

        for ii in range(num_hidden_layers + 1):
            # Input to hidden
            if ii == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            # Hidden to output
            elif ii == num_hidden_layers:
                self.layers.append(nn.Linear(hidden_dim, out_dim))
            # Hidden to hidden
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            #
            # Init weights with Xavier Glorot and set biases to zero
            #
            init.xavier_uniform_(self.layers[-1].weight)
            self.layers[-1].bias.data.fill_(0.0)

            self.add_module("layer_{}".format(ii), self.layers[-1])
            #
            # Batchnorm
            #
            if self.do_batchnorm and ii < num_hidden_layers:
                # Get out_features in a robust way by calling getattr
                lin = getattr(self, "layer_{}".format(ii))
                self.layers.append(nn.BatchNorm1d(lin.out_features))
                self.add_module("batchnorm_{}".format(ii), self.layers[-1])


    def forward(self, x):
        for jj in range(self.num_hidden_layers + 1):
            layer = getattr(self, "layer_{}".format(jj))
            x = layer(x)
            if jj < self.num_hidden_layers:
                x = self.act(x)

                # Batchnorm and/or dropout
                # warning is raised if both are selected in constructor
                if self.do_batchnorm:
                    bn = getattr(self, "batchnorm_{}".format(jj))
                    x = bn(x)

                if self.do_dropout:
                    x = self.dropout_layer(x)
        return x
# ================================================================
#
# Parent class generates MLPs for the vertex embedding
#   for both the whole-graph and one-graph classes
# (carry-over from previous implementations)
# ================================================================
class GinParent(nn.Module):
    def __init__(self, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim, vertices_are_onehot, other_mlp_parameters={}):
        """
        :param input_data_dim: Dimension of the vertex attributes
        :param num_agg_steps: K, the number of WL iterations.  The number of neighborhood aggregations
        :param vertex_embed_dim: Dimension of the `hidden' vertex attributes iteration to iteration
        :param mlp_num_hidden: Number of layers.  1 layer is sigmoid(Wx). 2 layers is Theta sigmoid(Wx)
        :param mlp_hidden_dim: Number of neurons in each layer
        :param vertices_are_onehot: Are the vertex features one-hot-encoded?  Boolean
        :param vertex_embed_only: We are only interested in the vertex embeddings at layer K.
                   ..not forming a graph-wide embedding
                   ..note: this is helpful for debug
        """
        assert num_agg_steps > 0, "Number of aggregation steps should be positive"
        assert isinstance(vertices_are_onehot, bool)

        super(GinParent, self).__init__()

        self.vertices_are_onehot = vertices_are_onehot
        self.input_data_dim = input_data_dim
        self.num_agg_steps = num_agg_steps
        self.vertex_embed_dim = vertex_embed_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Init layers for embedding
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.gin_layers = []
        #
        # If vertex attributes are one-hot, we don't need an MLP before summation in the first layer
        #
        if not vertices_are_onehot:
            logging.info("User indicated: Vertex attributes are NOT one hot")
            # We need an extra MLP for embedding the features
            self.gin_layers.append(MLP(in_dim=self.input_data_dim,
                                       hidden_dim=mlp_hidden_dim,
                                       out_dim=vertex_embed_dim,
                                       num_hidden_layers=mlp_num_hidden,
                                       other_mlp_parameters=other_mlp_parameters))
            self.add_module("raw_embedding_layer", self.gin_layers[-1])
            #
            # MLP after aggregation is different here, because of the input dimension
            #
            self.gin_layers.append(MLP(in_dim=vertex_embed_dim,
                                       hidden_dim=mlp_hidden_dim,
                                       out_dim=vertex_embed_dim,
                                       num_hidden_layers=mlp_num_hidden,
                                       other_mlp_parameters=other_mlp_parameters))

            self.add_module("agg_0", self.gin_layers[-1])
        else:
            logging.info("User indicated: Vertex attributes ARE one hot")

        for itr in range(num_agg_steps):
            if itr == 0 and vertices_are_onehot:
                self.gin_layers.append(MLP(in_dim=self.input_data_dim,
                                           hidden_dim=mlp_hidden_dim,
                                           out_dim=vertex_embed_dim,
                                           num_hidden_layers=mlp_num_hidden,
                                           other_mlp_parameters=other_mlp_parameters))
            # Assume all 'hidden' vertex features are of the same dim
            else:
                self.gin_layers.append(MLP(in_dim=vertex_embed_dim,
                                           hidden_dim=mlp_hidden_dim,
                                           out_dim=vertex_embed_dim,
                                           num_hidden_layers=mlp_num_hidden,
                                           other_mlp_parameters=other_mlp_parameters))

            self.add_module("agg_{}".format(itr), self.gin_layers[-1])
        #
        # Compute graph embedding dim (note it won't be used if
        #   we only want vertex embeds, but that's fine)
        self.graph_embed_dim = self.input_data_dim + vertex_embed_dim * num_agg_steps

# ========================================
class GinMultiGraph(GinParent):
    """
    Designed for graph classification 
    """
    def __init__(self, adjmat_list, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim, vertices_are_onehot, target_dim, epsilon_tunable=False, dense_layer_dropout=0.0, other_mlp_parameters={}):
        """
        Most parameters defined in the parent class

        :param adjmat_list: List of all adjmats to be considered
        Purpose: force input validation, but not saved to any variable.
        The user will enter the graphs in the dataset.  In principle, the graphs passed to
        initialize could be different than those used in the forward method; it is up 
        to the user to properly do input validation on all desired graphs
        
        This is NOT stored as a self object; rest easy we're not wasting memory

        :param target_dim: Dimension of the response variable (the target)

        :param epsilon_tunable: Do we make epsilon in equation 4.1 tunable
        :param dense_layer_dropout: Dropout to apply to the dense layer.
                                    In accordance with the GIN paper's experimental section
        """
        # Make sure all entered matrices are coo
        def is_coo(mat):
            return isinstance(mat, sps.coo.coo_matrix)

        # Make sure there are ones on the diagonal.
        def diags_all_one(mat):
            return np.array_equal(mat.diagonal(), np.ones(mat.shape[0]))

        assert all(list(map(is_coo, adjmat_list))), "All adjacency matrices must be scipy sparse coo"
        assert all(list(map(diags_all_one, adjmat_list))), "All adjacency matrices must have ones on the diag"
        assert isinstance(dense_layer_dropout, float), "Dense layer dropout must be a float in 0 <= p < 1"
        assert 0 <= dense_layer_dropout < 1, "Dense layer dropout must be a float in 0 <= p < 1"

        super(GinMultiGraph, self).__init__(input_data_dim=input_data_dim,
                                            num_agg_steps=num_agg_steps,
                                            vertex_embed_dim=vertex_embed_dim,
                                            mlp_num_hidden=mlp_num_hidden,
                                            mlp_hidden_dim=mlp_hidden_dim,
                                            vertices_are_onehot=vertices_are_onehot,
                                            other_mlp_parameters=other_mlp_parameters
                                            )

        self.target_dim = target_dim
        self.add_module("last_linear", nn.Linear(self.graph_embed_dim, target_dim))

        self.epsilon_tunable = epsilon_tunable

        logging.info("Dense layer dropout: {}".format(dense_layer_dropout))
        self.dense_layer_dropout = nn.Dropout(p=dense_layer_dropout)

        if epsilon_tunable:
            logging.info("User indicated: epsilon_tunable = True")
            logging.info("Epsilon_k WILL be LEARNED via backprop")
            logging.info("It is initialized to zero")

            self.epsilons = nn.ParameterList()
            for ll in range(num_agg_steps):
                epsilon_k = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.epsilons.append(epsilon_k)
        else:
            logging.info("User indicated: epsilon_tunable = False")
            logging.info("Epsilon_k WILL NOT be learned via backprop (and set to zero implicitly)")


    def construct_sparse_operator_tensors(self, sparse_adjmats):
        """ Construct the matrices needed to perform
            hidden layer updates (pre-MLP)

            :param: Sparse adjmat in a BATCH (thus different from the list passed to the constructor)

            :return: Adjacency: A sparse block-diagonal torch tensor, where the blocks
                     are adjmats

            :return Summation matrix: A matrix of ones and zeros such that
                                  matrix multiplication will effectively
                                  compute the row sums of chunks of a matrix B

                                  Because B will store the vertex embeddings
                                  for every vertex, for every graph.
                                  We want to compute the sums within a graph.

            Example:

            Suppose we had a two-node graph, a three-node graph, and another two-node
               our matrix would look like
                [1, 1, 0, 0, 0, 0, 0]
             S= [0, 0, 1, 1, 1, 0, 0]
                [0, 0, 0, 0, 0, 1, 1]

            Then we will do S @ B
        """
        assert isinstance(sparse_adjmats, list)
        #
        # ADJMAT:
        #
        # >  Make diagonal scipy sparse matrix of adjmats
        diag_mat = sps.block_diag(sparse_adjmats)
        #
        # turn it into a torch sparse tensor
        #
        rows, cols = sps.find(diag_mat)[0:2]  # indices of nonzero rows and cols
        indx_tens = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)], dim=0)
        vals_tens = torch.ones(len(rows))
        self.block_adj = torch.sparse.FloatTensor(indx_tens, vals_tens)  # One may think this should be an int tensor, but we cannot multiply ints with floats in PyTorch
        #
        # MAKE THE SUMMATION MATRIX
        #  >> non-zero indices since we will make a sparse matrix
        sum_mat_cols = list(range(self.block_adj.shape[0]))
        sum_mat_rows = []
        for iii in range(len(sparse_adjmats)):
            num_nodes = sparse_adjmats[iii].shape[0]
            sum_mat_rows.extend([iii for jjj in range(num_nodes)])

        sum_indx = torch.stack([torch.LongTensor(sum_mat_rows), torch.LongTensor(sum_mat_cols)], dim=0)
        sum_vals = torch.ones(sum_indx.shape[1])
        self.sum_tensor = torch.sparse.FloatTensor(sum_indx, sum_vals)

    def forward(self, adjmat_list, X):
        """
        Get a graph-level prediction for a list of graphs
        :param X: Vertex attributes for every vertex in every batch
        :param adjmat_list: List of adjacency matrices in batch
        """
        # check that #vertices and X dimension coincide
        total_vertices = np.sum([mat.shape[0] for mat in adjmat_list])
        assert total_vertices == X.shape[0], "Total vertices must match the number of rows in X"
        assert X.shape[1] == self.input_data_dim, "Number of columns in X must match self.input_data_dim"

        # Construct matrices that will allow vectorized operations of
        # "sum neighbors" and "sum all vertices within a graph"
        self.construct_sparse_operator_tensors(adjmat_list)

        # Get embedding from X
        self.graph_embedding = torch.mm(self.sum_tensor, X)

        if not self.vertices_are_onehot:
            embedding = getattr(self, "raw_embedding_layer")
            H = embedding(X)
        else:
            H = X.clone()

        for kk in range(self.num_agg_steps):
            # Sum self and neighbor
            if not self.epsilon_tunable:
                # Aggregation in matrix form: (A + I)H
                agg_pre_mlp = torch.mm(self.block_adj, H)
                # print(agg_pre_mlp)
            else:
                #
                # Add epsilon to h_v, as in equation 4.1
                # Note that the proper matrix multiplication is
                # (A + (1+epsilon)I)H = (A+I)H + epsilon H
                #
                # Our implementation avoids making epsilon interact with the
                #  adjacency matrix, which would make PyTorch want to
                #  track gradients through the adjmat by default
                #
                epsilon_k = self.epsilons[kk]
                agg_pre_mlp = torch.mm(self.block_adj, H) + epsilon_k*H


            mlp = getattr(self, "agg_{}".format(kk))
            H = mlp(agg_pre_mlp)
            #
            layer_k_embed = torch.mm(self.sum_tensor, H)
            self.graph_embedding = torch.cat((self.graph_embedding,
                                              layer_k_embed),
                                             dim=1)
            #
        last_layer = getattr(self, "last_linear")
        final = last_layer(self.graph_embedding)

        # apply dropout and return (note dropout is 0.0 by default)
        return self.dense_layer_dropout(final)

# ========================================
#
# RP-GIN.  Use GIN as \harrow{f} in
#          relational pooling model.
# ========================================
class RpGin(GinMultiGraph):
    """
    Wrap GIN in relational pooling.
    Here we randomly permute the adjacency matrix (while preserving isomorphic invariance)
    A_new = P^T @ A @ P
    where @ denotes matrix multiplication and P is a permutation matrix.

    We then forward the shuffled mats (see paper for theoretical explanation)
    """
    def __init__(self, adjmat_list, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim, target_dim, featureless_case, vertices_are_onehot=False, epsilon_tunable = False, dense_layer_dropout = 0.0, other_mlp_parameters = {}):
        """ Parameters are defined in parent class
            :param: featureless_case: bool, is the input featureless?
            if featureless, input_data_dim is used as the largest expected graph
        """
        assert isinstance(featureless_case, bool)
        self.featureles_case = featureless_case
        if featureless_case:
            logging.info("User has indicated that the graphs are featureless")
            logging.info("The number of vertices in the largest expected graph is {}".format(input_data_dim))
            self.featureles_case = True
        else:
            self.featureles_case = False
            raise NotImplementedError("Have only considered featureless case thus far. Set input dim to 0 for featureless case")

        super(RpGin, self).__init__(adjmat_list, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim, vertices_are_onehot, target_dim, epsilon_tunable, dense_layer_dropout, other_mlp_parameters)

    def permute_adjmat(self, mat):
        """
        :param mat: A scipy sparse matrix representing an adjacency matrix
        :return: A permuted matrix corresponding to an isomorphic graph, ie
        P^T @ mat @ P
        where @ denotes matrix multiplication and P is a permutation matrix.
        """
        # form a permutation matrix by shuffling an identity matrix
        #  the result of sparse_shuffle will be compressed row, need to coo it
        P = sps.coo_matrix(sparse_shuffle(sps.eye(mat.shape[0])))
        return P.transpose() @ mat @ P

    def forward(self, sparse_adjmats, X):
        """
        :param sparse_adjmats: List of adjacency  matrices for the graphs in the batch
        :param X: Vertex features
        :return:
        """
        # (1) Permute all the adjacency matrices in the list
        # (2) Forward to GIN
        if self.featureles_case:
            return super(RpGin, self).forward(adjmat_list=list(map(self.permute_adjmat, sparse_adjmats)),
                                              X=X)
        else:
            pass

    def inference(self, sparse_adjmats, X, num_inf_perms=5):
        """
        To do proper inference, we sample multiple
        permutations and average
        :param num_inf_perms: Number of random permutations to do at inference time
        """
        divisor = (1.0/num_inf_perms)
        preds = divisor * self.forward(sparse_adjmats, X)
        for iii in range(num_inf_perms-1):
            preds += divisor * self.forward(sparse_adjmats, X)

        return preds
