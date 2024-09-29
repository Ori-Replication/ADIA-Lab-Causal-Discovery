from pandas import read_csv
from matplotlib.pyplot import subplots, show
import pickle
import networkx as nx
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

causal_graph = read_csv("./adjacency_matrix.csv").to_numpy()

with open(r"ground_truth_G.pkl", "rb") as input_file_G:
    ground_truth = pickle.load(input_file_G)

# Generate figure and axis
fig, (ax1, ax2) = subplots(1,2, figsize=(12,5))

cm1 = ax1.pcolormesh(nx.to_numpy_array(ground_truth))
ax1.set_title("Ground Truth", fontsize=20)
ax1.set_xlabel("Vertices", fontsize=18)
ax1.set_ylabel("Vertices", fontsize=18)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax1_divider = make_axes_locatable(ax1)
ax1.grid(visible=True)
cax1 = ax1_divider.append_axes("right", size="7%", pad="4%")
cb1 = fig.colorbar(cm1, cax=cax1)

cm2 = ax2.pcolormesh(causal_graph)
ax2.set_title("Recovered Graph", fontsize=20)
ax2.set_xlabel("Vertices", fontsize=18)
ax2.set_ylabel("Vertices", fontsize=18)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax2_divider = make_axes_locatable(ax2)
ax2.grid(visible=True)
cax2 = ax2_divider.append_axes("right", size="7%", pad="4%")
cb2 = fig.colorbar(cm2, cax=cax2)

fig.tight_layout() # Fit everything nicely in figure
show()
