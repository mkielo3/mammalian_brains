"""
Calculates the Smoothness of an SOM by modality
"""

from numbasom import lattice_activations
import numpy as np
import pandas as pd
import pysal.lib as ps
from esda.moran import Moran, Moran_Local
import altair as alt
import sys
from io import StringIO
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')


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
    old_stdout = sys.stdout
    sys.stdout = StringIO() # stop standard out bc lattice_activations forces a print
    full_rf = calculate_receptive_fields(data)
    som_size = data['args'].som_size[0]
    value_list = []
    for x in range(som_size):
        for y in range(som_size):
            source = full_rf[(full_rf['som_x'] == x) & (full_rf['som_y'] == y)].set_index(['patch_x', 'patch_y'])[['value']].groupby(['patch_x', 'patch_y']).mean().rank(pct=True)
            source.loc[source['value'] < 0.9, 'value'] = 0.9
            true_value = calc_morans_i(source['value'].unstack().fillna(0.9).values)
            value_list.append(true_value)
    sys.stdout = old_stdout
    return np.mean(value_list), full_rf




def draw_olfaction_rf(data):
	c_list = []
	_, df = calculate_average_rf_locality(data)

	for i, row in df[['som_x', 'som_y']].drop_duplicates().sample(5).iterrows():
		x, y = row['som_x'], row['som_y']

		source = df[(df['som_x'] == x) & (df['som_y'] == y)].groupby([
				df['patch_x'],
				df['patch_y']
			])[['value']].mean().reset_index()

		# show first 100 channels for viz purposes
		source['value'] = source['value'].iloc[0:100].rank(pct=True) 
		source.loc[source['value'] < 0.9, 'value'] = 0.9
		chart = alt.Chart(source).mark_rect(
			width=5,
			height=25,
			stroke=None,
			strokeWidth=0
		).encode(
			x=alt.X('patch_x:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			y=alt.Y('patch_y:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno'), legend=None)
		).properties(
			width=225,
			height=40
		)
		c_list.append(chart)

	return alt.vconcat(*c_list, spacing=10).properties(
		title=alt.TitleParams(
			text="Olfaction",
			anchor="middle",
			fontSize=16
		))


def draw_audio_rf(data):
	_, df = calculate_average_rf_locality(data)

	c_list = []
	som_size = data['lattice'].shape[0]
	x_range = range(5) if som_size == 5 else list(range(3, 23, 4)) # don't pick a random index if size 5 because of the redundancy
	print (x_range)
	for x in x_range:

		source = df[(df['som_x'] == x) & (df['som_y'] == 0)].groupby([
				df['patch_x'],
				df['patch_y']
			])[['value']].mean().reset_index()

		# show first 100 channels for viz purposes
		if som_size == 25:
			source['value'] = source['value'].rank(pct=True)
			source.loc[source['value'] < 0.9, 'value'] = 0.9
		elif som_size == 5:
			pass

		chart = alt.Chart(source).mark_rect(
			width=5,
			height=25,
			stroke=None,
			strokeWidth=0
		).encode(
			x=alt.X('patch_x:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			y=alt.Y('patch_y:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno'), legend=None)
		).properties(
			width=225,
			height=40
		)
		c_list.append(chart)

	return alt.vconcat(*c_list, spacing=10).properties(
		title=alt.TitleParams(
			text="Audition",
			anchor="middle",
			fontSize=16
		)).resolve_scale(color='independent')



def draw_tem_tf(data):
	_, df = calculate_average_rf_locality(data)

	c_list = []
	for i, row in df[['som_x', 'som_y']].drop_duplicates().sample(5).iterrows():
		x, y = row['som_x'], row['som_y']
		
		source = df[(df['som_x'] == x) & (df['som_y'] == y)].groupby([
				df['patch_x'],
				df['patch_y']
			])[['value']].mean().reset_index()

		source['value'] = source['value'].rank(pct=True, method='average')
		source.loc[source['value'] < 0.9, 'value'] = 0.9

		chart = alt.Chart(source).mark_rect(
			width=5,
			height=25,
			stroke=None,
			strokeWidth=0
		).encode(
			x=alt.X('patch_x:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			y=alt.Y('patch_y:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno'), legend=None)
		).properties(
			width=225,
			height=40
		)
		c_list.append(chart)

	return alt.vconcat(*c_list, spacing=10).properties(
		title=alt.TitleParams(
			text="TEM",
			anchor="middle",
			fontSize=16
		))


def draw_vision_rf(data):
	_, df = calculate_average_rf_locality(data)
	som_size = data['lattice'].shape[0]

	c_list = []
	for i, row in df[['som_x', 'som_y']].drop_duplicates().sample(9).iterrows():
		x, y = row['som_x'], row['som_y']

		source = df[(df['som_x'] == x) & (df['som_y'] == y)].groupby([
				df['patch_x'].floordiv(10),
				df['patch_y'].floordiv(10)
			])[['value']].mean().reset_index()

		if som_size > 5:
			source['value'] = source['value'].rank(pct=True, method='average')
			source.loc[source['value'] < 0.9, 'value'] = 0.9

		chart = alt.Chart(source).mark_rect(
			width = 6 if som_size == 25 else 7,
			height = 6 if som_size == 25 else 7,
			stroke=None,
			strokeWidth=0
		).encode(
			x=alt.X('patch_x:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			y=alt.Y('patch_y:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno'), legend=None)
		).properties(
			width=80,
			height=80
		)
		c_list.append(chart)

	grid = alt.vconcat(
		alt.hconcat(*c_list[0:3], spacing=1),
		alt.hconcat(*c_list[3:6], spacing=1),
		alt.hconcat(*c_list[6:9], spacing=1),
		spacing=1
	).properties(
		title=alt.TitleParams(
			text="Vision",
			anchor="middle",
			fontSize=16
		)
	)

	return grid


def draw_touch_rf(data):
	_, df = calculate_average_rf_locality(data)
	som_size = data['lattice'].shape[0]

	c_list = []
	for i, row in df[['som_x', 'som_y']].drop_duplicates().sample(9).iterrows():
		x, y = row['som_x'], row['som_y']

		source = df[(df['som_x'] == x) & (df['som_y'] == y)].groupby([
				df['patch_x'],
				df['patch_y']
			])[['value']].mean().reset_index()

		if som_size > 5:
			source['value'] = source['value'].rank(pct=True)
			source.loc[source['value'] < 0.9, 'value'] = 0.9

		source = (source.set_index(['patch_x', 'patch_y'])['value']+1e-9).unstack().reindex(index=range(*data['x_range']), columns=range(*data['y_range'])).fillna(0).stack().reset_index().replace(0, np.nan)
		source.columns=['patch_x', 'patch_y', 'value']

		chart = alt.Chart(source).mark_rect(
			width=5,
			height=5,
			stroke=None,
			strokeWidth=0
		).encode(
			x=alt.X('patch_x:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			y=alt.Y('patch_y:O',
				axis=alt.Axis(ticks=False, labels=False, title=None, domain=False, grid=False),
				scale=alt.Scale(paddingInner=0)
			),
			color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno'), legend=None)
		).properties(
			width=80,
			height=80
		)
		c_list.append(chart)

	grid = alt.vconcat(
		alt.hconcat(*c_list[0:3], spacing=1),
		alt.hconcat(*c_list[3:6], spacing=1),
		alt.hconcat(*c_list[6:9], spacing=1),
		spacing=1
	).properties(
		title=alt.TitleParams(
			text="Somatosensation",
			anchor="middle",
			fontSize=16
		)
	)

	return grid


def plot_rf(experiment_name, modality_list, load_pickle_output):
	c_list = []
	for modality in modality_list:
		data = load_pickle_output(experiment_name, modality.modality)
		if modality.modality == 'olfaction':
			c_list.append(draw_olfaction_rf(data))
		elif modality.modality == 'touch':
			c_list.append(draw_touch_rf(data))
		elif modality.modality == 'vision':
			c_list.append(draw_vision_rf(data))
		elif modality.modality == 'audio':
			c_list.append(draw_audio_rf(data))
		elif modality.modality == 'memory':
			c_list.append(draw_tem_tf(data))
	return alt.hconcat(*c_list).resolve_scale(color='independent').configure_view(strokeWidth=0)