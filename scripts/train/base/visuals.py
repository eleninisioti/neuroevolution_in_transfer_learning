import copy
import os
import matplotlib.pyplot as plt
import seaborn
import jax.numpy as jnp
from sklearn.decomposition import PCA
from bokeh.plotting import figure, output_file, save
import pandas as pd
import networkx as nx
import numpy as onp
from bokeh.plotting import figure
from bokeh.palettes import Iridescent, Blues8
from bokeh.plotting import from_networkx
from bokeh.models import Arrow, VeeHead, NormalHead, MultiLine, Bezier
from bokeh.models import Legend, Circle, ColumnDataSource, HoverTool, LabelSet
from bokeh.transform import linear_cmap
import copy
from matplotlib.colors import LinearSegmentedColormap
from bokeh.io import export_png



# ----- general figure configuration -----
custom_palette = ["#ff595e", "#1982c4","#8ac926", "#ffca3a","#6a4c93"]

params = {'legend.fontsize': 9,
          "figure.autolayout": True,
          'font.size': 9,
          "axes.prop_cycle": plt.cycler(color=custom_palette)}
plt.rcParams.update(params)
cm = 1 / 2.54 # convert cm to inches
fig_size = (10*cm , 5*cm) # (3,2) centimeters


divergent_palette = ["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6","#ee9b00","#ca6702","#bb3e03","#ae2012","#9b2226"]
divergent_cmap = LinearSegmentedColormap.from_list("custom_divergent", divergent_palette, N=256)

def create_color_vector(pca_values, colormap, min_value, max_value):
    # Normalize the PCA values to a range between 0 and 255 (for the colormap)
    normalized_pca = (pca_values - min_value) / (max_value - min_value)
    normalized_pca = (normalized_pca * (len(colormap) - 1)).astype(int)

    # Map normalized PCA values to the colormap
    color_vector = [colormap[i] for i in normalized_pca]
    return color_vector





    # save heatmap of weights
    fig, ax = plt.subplots(figsize=fig_size)  # Adjust figure size here

    if len(weights.shape)==1:
        weights = weights.reshape(-1,1)
    try:
        heatmap = seaborn.heatmap(weights, cmap=divergent_cmap, fmt='.1e', cbar=True,annot=annotate)
    except ValueError:
        print("Error when generating heatmap")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_offset_position('left')
    ax.set_title("Policy network")
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(filename + ".png", format='png', dpi=300)
    plt.clf()
    plt.close()

def viz_heatmap(weights, filename, annotate=False):



    # save heatmap of weights
    fig, ax = plt.subplots(figsize=fig_size)  # Adjust figure size here

    if len(weights.shape)==1:
        weights = weights.reshape(-1,1)
    try:
        heatmap = seaborn.heatmap(weights, cmap=divergent_cmap, fmt='.1e', cbar=True,annot=annotate)
    except ValueError:
        print("Error when generating heatmap")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_offset_position('left')
    ax.set_title("Policy network")
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(filename + ".png", format='png', dpi=300)
    plt.clf()
    plt.close()
    
    

def viz_histogram(values, filename, feature_name="Reward"):
    fig, ax = plt.subplots()
    values = onp.array(values)
    values[values == -onp.inf] = 0

    output = ax.hist(values, color="#005F73")
    print(output)
    plt.xlabel(feature_name)
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=300)
    path = filename + ".png"

    return path





