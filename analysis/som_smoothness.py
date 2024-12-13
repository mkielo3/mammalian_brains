import numpy as np
import pandas as pd
import altair as alt
from scipy.interpolate import RegularGridInterpolator
import random

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return np.array(rgb) / 255.0

def rgb_to_hex(rgb_arr):
    rgb_255 = (np.clip(rgb_arr, 0, 1) * 255).astype(int)
    return f'#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}'

def compute_distance(data, norm_factor):
    "Calculate Distance Betwee Neighbors for SOM Smoothness"
    modality = data['modality']
    flat_som = pd.DataFrame(data['closest']).T

    if modality in ['vision', 'touch']:
        dim0 = flat_som.map(lambda x: x[0]).unstack()
        dim1 = flat_som.map(lambda x: x[1]).unstack()
        dist = np.sqrt(
            np.mean([
                np.mean((dim0 - dim0.shift(s1).T.shift(s2).T) ** 2) +
                np.mean((dim1 - dim1.shift(s1).T.shift(s2).T) ** 2)
                for s1 in [-1, 0, 1]
                for s2 in [-1, 0, 1]
                if not (s1 == 0 and s2 == 0)
            ])
        ) / norm_factor
    else:
        dim0 = flat_som[0].map(lambda x: x[0]).unstack()
        if modality == 'audio':
            dist = np.sqrt(
                np.mean([
                    np.mean((dim0 - dim0.shift(s1)) ** 2)
                    for s1 in [-1, 0, 1]
                    if s1 != 0
                ])
            ) / norm_factor
        else:
            dist = np.sqrt(
                np.mean([
                    np.mean((dim0 - dim0.shift(s1).T.shift(s2).T) ** 2)
                    for s1 in [-1, 0, 1]
                    for s2 in [-1, 0, 1]
                    if not (s1 == 0 and s2 == 0)
                ])
            ) / norm_factor
    return dist

def gen_norm_factor(data, t=10000):
    "Calculate Expected Smoothness for Random Map by Modality"
    modality = data['modality']
    modalities = {
        'vision': {'w': data['args'].vision_patch_size, 'dim_keys': ['x_range', 'y_range']},
        'touch': {'w': data['args'].touch_patch_size, 'dim_keys': ['x_range', 'y_range']},
        'olfaction': {'w': data['args'].olfaction_patch_size, 'dim_keys': ['x_range']},
        'audio': {'w': data['args'].audio_patch_size, 'dim_keys': ['x_range']},
        'memory': {'w': data['args'].tem_patch_size, 'dim_keys': ['x_range']}
    }
    
    params = modalities[modality]
    w = params['w']
    dim_list = [range(data[key][0], data[key][1] + w, w) for key in params['dim_keys']]
    dist_list = []
    n = len(data['closest'])
    
    for _ in range(t):
        tot_dist = 0
        for dim in dim_list:
            reference_df = pd.DataFrame(data['closest']).T
            reference_df[0] = random.choices(dim, k=n)
            flat_som = reference_df[0].unstack()
            tot_dist += np.mean([
                np.mean((flat_som - flat_som.shift(s1).T.shift(s2).T) ** 2)
                for s1 in [-1, 0, 1]
                for s2 in [-1, 0, 1]
                if not (s1 == 0 and s2 == 0)
            ])
        dist_list.append(np.sqrt(tot_dist))
    return np.mean(dist_list)

def interpolate_colormap(output_size=(50, 50)):
    "2d viridis colormap"

    hex_colors = np.array([
        ["#440154FF", "#481567FF", "#482677FF", "#453781FF", "#404788FF"],
        ["#39568CFF", "#33638DFF", "#2D708EFF", "#287D8EFF", "#238A8DFF"],
        ["#1F968BFF", "#20A387FF", "#29AF7FFF", "#3CBB75FF", "#55C667FF"],
        ["#73D055FF", "#95D840FF", "#B8DE29FF", "#DCE319FF", "#FDE725FF"]
    ]).T
    hex_colors_5x4 = np.flipud(hex_colors)
    
    colors_5x4 = np.zeros((5, 4, 3))
    for i in range(5):
        for j in range(4):
            colors_5x4[i, j] = hex_to_rgb(hex_colors_5x4[i, j])
    
    x_orig = np.linspace(0, 1, 4)
    y_orig = np.linspace(0, 1, 5)
    
    interpolators = []
    for channel in range(3):
        interpolator = RegularGridInterpolator(
            (y_orig, x_orig),
            colors_5x4[:, :, channel],
            method='cubic',
            bounds_error=False,
            fill_value=None
        )
        interpolators.append(interpolator)
    
    y_fine = np.linspace(0, 1, output_size[0])
    x_fine = np.linspace(0, 1, output_size[1])
    Y_fine, X_fine = np.meshgrid(y_fine, x_fine, indexing='ij')
    points = np.stack([Y_fine.flatten(), X_fine.flatten()], axis=1)
    
    interpolated_colors = np.zeros((output_size[0] * output_size[1], 3))
    for channel in range(3):
        interpolated_colors[:, channel] = interpolators[channel](points)
    
    interpolated_colors = interpolated_colors.reshape(output_size[0], output_size[1], 3)
    interpolated_colors = np.clip(interpolated_colors, 0, 1)
    
    hex_output = np.empty(output_size, dtype='U7')
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            hex_output[i, j] = rgb_to_hex(interpolated_colors[i, j])
    
    return hex_output

def gen_charts(data):
    "Generate Plots"

    modality = data['modality']
    som_size = data['args'].som_size
    source = pd.DataFrame(data['closest']).loc[0].reset_index().rename(columns={'level_0': 'y', 'level_1': 'x', 0: 'c_raw'})

    if modality in ['olfaction', 'audio']:
        source['z'] = source['c_raw'].map(lambda x: x[0])
        scale = alt.Scale(domain=data['x_range'], scheme='viridis')
        index_color = alt.Color('z:Q', scale=scale, legend=None)
        if modality == 'olfaction':
            index_df = pd.DataFrame([(x, y, x) for x in range(*data['x_range'], 10) for y in range(*data['x_range'], 10)], columns=['x', 'y', 'z'])
            title_str = 'Olfaction'
        else:
            index_df = pd.DataFrame([(y, x, x) for x in range(*data['x_range'], 10) for y in range(*data['x_range'], 10)], columns=['x', 'y', 'z'])
            title_str = 'Audio'
    elif modality in ['vision', 'touch']:
        xdim = data['x_range'][1] + 1
        ydim = data['y_range'][1] + 1
        interpolated = interpolate_colormap(output_size=(xdim, ydim))
        source['z'] = source['c_raw'].map(lambda x: interpolated[x[0], x[1]])
        index_color = alt.Color('z:N', scale=None, legend=None)
        if modality == 'vision':
            index_df = pd.DataFrame([(x, y, interpolated[x, y]) for x in range(*data['x_range'], 3) for y in range(*data['y_range'], 3)], columns=['x', 'y', 'z'])
            title_str = 'Vision'
        else:
            index_df = pd.DataFrame([(x, y, interpolated[x, y]) for x in range(*data['x_range'], 1) for y in range(*data['y_range'], 1)], columns=['x', 'y', 'z'])
            title_str = 'Somatosensation'
    elif modality == 'memory':
        source['z'] = source['c_raw'].map(lambda x: x[0])
        scale = alt.Scale(domain=data['x_range'], scheme='viridis')
        index_color = alt.Color('z:Q', scale=scale, legend=None)
        title_str = 'Relational Memory'

    else:
        assert (False)

    som_chart = alt.Chart(source).mark_rect(
        width=6,
        height=6,
        stroke=None, strokeWidth=0
    ).encode(
        x=alt.X('x:O', axis=alt.Axis(ticks=False, labels=False, title=None), scale=alt.Scale(paddingInner=0)),
        y=alt.Y('y:O', axis=alt.Axis(ticks=False, labels=False, title=None), scale=alt.Scale(paddingInner=0)),
        color=index_color
    ).properties(
        width=140,
        height=140,
        title=alt.TitleParams(title_str, anchor='middle', fontSize=16)
    )

    # index_chart = alt.Chart(index_df).mark_rect(
    #     width=6,
    #     height=6,
    # ).encode(
    #     x=alt.X('x:O', axis=alt.Axis(ticks=False, labels=False, title=None)),
    #     y=alt.Y('y:O', axis=alt.Axis(ticks=False, labels=False, title=None)),
    #     color=index_color
    # ).properties(
    #     width=150,
    #     height=150,
    #     title=alt.TitleParams(title_str, anchor='middle', fontSize=16)
    # )

    return som_chart, None, source
