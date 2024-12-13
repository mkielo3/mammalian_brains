""" Train the Wang et al model, forked from https://github.com/gyyang/olfaction_evolution/blob/master/notebooks/quickstart.ipynb """

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from torch import nn
import time
import numpy as np
import math


def euclidean_distances(matrix1, matrix2):
    squared_distances = (
        np.sum(matrix1**2, axis=1, keepdims=True)
        + np.sum(matrix2**2, axis=1)
        - 2 * np.dot(matrix1, matrix2.T)
    )

    squared_distances = np.maximum(squared_distances, 0)
    return np.sqrt(squared_distances)


def get_labels(prototypes, odors):
    """Get label of nearest prototype for odors."""
    dist = euclidean_distances(prototypes, odors)
    return np.argmin(dist, axis=0)


def _get_normalization(norm_type, num_features=None):
    if norm_type is not None:
        if norm_type == 'batch_norm':
            return nn.BatchNorm1d(num_features)
    return lambda x: x


class Layer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix can be set non-negative
    """
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                sign_constraint=False,
                weight_initializer=None,
                weight_initial_value=None,
                bias_initial_value=0,
                pre_norm=None,
                post_norm=None,
                dropout=False,
                dropout_rate=None,
                ):
        super(Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_initializer = weight_initializer
        if weight_initial_value:
            self.weight_init_range = weight_initial_value
        else:
            self.weight_init_range = 2. / in_features
        self.bias_initial_value = bias_initial_value
        self.sign_constraint = sign_constraint
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.pre_norm = _get_normalization(pre_norm, num_features=out_features)
        self.activation = nn.ReLU()
        self.post_norm = _get_normalization(post_norm, num_features=out_features)

        if dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        if self.sign_constraint:
            self._reset_sign_constraint_parameters()
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _reset_sign_constraint_parameters(self):
        if self.weight_initializer == 'constant':
            init.constant_(self.weight, self.weight_init_range)
        elif self.weight_initializer == 'uniform':
            init.uniform_(self.weight, 0, self.weight_init_range)
        elif self.weight_initializer == 'normal':
            init.normal_(self.weight, 0, self.weight_init_range)
        else:
            raise ValueError('Unknown initializer', str(self.weight_initializer))

        if self.bias is not None:
            init.constant_(self.bias, self.bias_initial_value)

    @property
    def effective_weight(self):
        if self.sign_constraint:
            weight = torch.abs(self.weight)
        else:
            weight = self.weight

        return weight

    def forward(self, input):
        weight = self.effective_weight
        pre_act = F.linear(input, weight, self.bias)
        pre_act_normalized = self.pre_norm(pre_act)
        output = self.activation(pre_act_normalized)
        output_normalized = self.post_norm(output)
        output_normalized = self.dropout(output_normalized)
        return output_normalized


class FullModel(nn.Module):
    """"The full 3-layer model."""
    def __init__(self, c_class, n_orn):
        super(FullModel, self).__init__()
        # ORN-PN
        self.layer1 = Layer(n_orn, 50,
                            weight_initializer='normal',
                            sign_constraint=True,
                            pre_norm='batch_norm',
                            )

        # PN-KC
        self.layer2 = Layer(50, 2500,
                            weight_initializer='uniform',
                            weight_initial_value=0.2,
                            bias_initial_value=-1,
                            sign_constraint=True,
                            dropout=True,
                            dropout_rate=0.5)

        self.layer3 = nn.Linear(2500, c_class)  # KC-output
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        act1 = self.layer1(x)
        act2 = self.layer2(act1)
        y = self.layer3(act2)
        if target is None:
            return y
        loss = self.loss(y, target)
        with torch.no_grad():
            _, pred = torch.max(y, 1)
            acc = (pred == target).sum().item() / target.size(0)
        return {'loss': loss, 'acc': acc, 'kc': act2}

    @property
    def w_orn2pn(self):
        # Transpose to be consistent with tensorflow default
        return self.layer1.effective_weight.data.cpu().numpy().T

    @property
    def w_pn2kc(self):
        return self.layer2.effective_weight.data.cpu().numpy().T



def load_and_train_model(DEVICE, epochs=15, n_orn=50, n_class=100):
    """ Train the Wang et al model, copied from https://github.com/gyyang/olfaction_evolution/blob/master/notebooks/quickstart.ipynb 
        n_orn = number of olfactory receptor neurons
        n_class = number of classes 
    """
        
    # Dataset
    n_train = 1000000  # number of training examples
    n_val = 10000  # number of validation examples

    prototypes = np.random.uniform(0, 1, (n_class, n_orn)).astype(np.float32)
    train_x = np.random.uniform(0, 1, (n_train, n_orn)).astype(np.float32)
    val_x = np.random.uniform(0, 1, (n_val, n_orn)).astype(np.float32)

    train_y = get_labels(prototypes, train_x).astype(np.int32)
    val_y = get_labels(prototypes, val_x).astype(np.int32)
    batch_size = 256

    model = FullModel(n_class, n_orn)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_data = torch.from_numpy(train_x).float().to(DEVICE)
    train_target = torch.from_numpy(train_y).long().to(DEVICE)
    val_data = torch.from_numpy(val_x).float().to(DEVICE)
    val_target = torch.from_numpy(val_y).long().to(DEVICE)

    loss_train = 0
    res = {'acc': np.nan}
    total_time, start_time = 0, time.time()

    for _ in range(epochs):
        with torch.no_grad():
            model.eval()
            res_val = model(val_data, val_target)

        model.train()
        random_idx = np.random.permutation(n_train)
        idx = 0
        while idx < n_train:
            batch_indices = random_idx[idx:idx+batch_size]
            idx += batch_size

            res = model(train_data[batch_indices],
                        train_target[batch_indices])
            optimizer.zero_grad()
            res['loss'].backward()
            optimizer.step()

    print('Train/Validation accuracy {:0.2f}/{:0.2f}'.format(res['acc'], res_val['acc']))
    model.eval()    
    return model, val_data # we care about KC-output, which is the last layer
