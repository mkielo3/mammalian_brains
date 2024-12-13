"""
Calculates the Smoothness of an SOM by modality
"""

from numbasom import lattice_activations
import numpy as np
import pandas as pd
import pysal.lib as ps
from esda.moran import Moran, Moran_Local

def calculate_receptive_fields(data):
    flat_activations = []
    for (i, x) in data['activations']:
        acts = lattice_activations(np.array(x.unsqueeze(dim=0)), data['lattice'])[0]
        for ii in range(acts.shape[0]):
            for jj in range(acts.shape[1]):
                flat_activations.append([i[0], i[1], ii, jj, acts[ii][jj]])

    activation_df = pd.DataFrame(flat_activations, columns=['patch_x', 'patch_y', 'som_x', 'som_y', 'value'])
    return activation_df


def calc_morans_i(x):
    rows, cols = x.shape
    w = ps.weights.lat2W(rows, cols)
    xx = x.flatten()
    moran = Moran(xx, w)
    return moran.I


def calculate_average_rf_locality(data):
    full_rf = calculate_receptive_fields(data)
    som_size = data['args'].som_size[0]
    value_list = []
    for x in range(som_size):
        for y in range(som_size):
            source = full_rf[(full_rf['som_x'] == x) & (full_rf['som_y'] == y)].set_index(['patch_x', 'patch_y'])[['value']].groupby(['patch_x', 'patch_y']).mean().rank(pct=True)
            source.loc[source['value'] < 0.9, 'value'] = 0.9
            true_value = calc_morans_i(source['value'].unstack().fillna(0.9).values)
            value_list.append(true_value)
    return np.mean(value_list), full_rf
