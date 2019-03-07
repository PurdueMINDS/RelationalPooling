import torch
def construct_onehot_ids(graph_size, onehot_dim):
    """Assign one hot identifiers of dimension onehot_dim
    :return: A matrix of dim (graph_size by onehot_dim)
             where each row is a one-hot identifier
             E.g. if onehot_dim = 3
             [1, 0, 0]
             [0, 1, 0]
             [0, 0, 1]
             .   .  .
             .   .  .
             .   .  .
    """
    ID = torch.zeros(graph_size, onehot_dim)
    col = 0
    for row in range(graph_size):
        ID[row, col] = 1
        col = (col + 1) % onehot_dim

    return ID