import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np

nodes = pd.read_excel('nodes.xlsx', usecols=['NodeId', 'Labels'])
edges = pd.read_excel('edges.xlsx', usecols=['sourceNodeId', 'targetNodeId'])

G = nx.from_pandas_edgelist(edges, 'sourceNodeId', 'targetNodeId')

infection_prob = {}

for node in G.nodes():
    neighbors = list(G.neighbors(node))
    infected_neighbors = [
        n for n in neighbors if nodes[nodes['NodeId'] == n]['Labels'].values[0] == 'L1']
    infection_prob[node] = len(infected_neighbors) / len(neighbors)


prob_values = np.array(list(infection_prob.values()))
Marz_jodasazi = np.percentile(prob_values, 0)
Marz_jodasazi = 0

high_risk_nodes = [node for node,
                   prob in infection_prob.items() if prob > Marz_jodasazi]

color_map = []
for node in G:
    if nodes[nodes['NodeId'] == node]['Labels'].values[0] == 'L1':
        color_map.append('red')
    elif node in high_risk_nodes:
        color_map.append('yellow')
    else:
        color_map.append('cyan')

nx.draw(G, node_color=color_map, with_labels=False, node_size=100)
plt.show()
