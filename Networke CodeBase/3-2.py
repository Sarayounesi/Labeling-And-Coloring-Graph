import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from networkx.algorithms import community
from networkx.algorithms.community.centrality import girvan_newman

nodes = pd.read_excel('nodes.xlsx', usecols=['NodeId', 'Labels'])
edges = pd.read_excel('edges.xlsx', usecols=['sourceNodeId', 'targetNodeId'])

G = nx.from_pandas_edgelist(edges, 'sourceNodeId', 'targetNodeId')

# communities_generator = nx.community.girvan_newman(G)
# top_level_communities = next(communities_generator)
# next_level_communities = next(communities_generator)
# sorted(map(sorted, next_level_communities))

c = nx.community.greedy_modularity_communities(G)
a = sorted(c[0])
b = sorted(c[1])
c = sorted(c[2])
print("--------------------------------------------------------------------------------------------------------------")

print("Modularity_communities [0]", a)

print("--------------------------------------------------------------------------------------------------------------")
print("Modularity_communities [1]", a)
print("--------------------------------------------------------------------------------------------------------------")

print("Modularity_communities [2]", a)
print("--------------------------------------------------------------------------------------------------------------")


print("kernighan_lin_bisection", nx.kernighan_lin_bisection(
    G, partition=None, max_iter=10, weight='weight', seed=None))

# communities_generator = nx.community.girvan_newman(G)
# top_level_communities = next(communities_generator)
# next_level_communities = next(communities_generator)


# communities_generator = community.girvan_newman(G)
# print('5.girvan_newman', tuple(sorted(c) for c in next(communities_generator)))

# print("--------------------------------------------------------------------------------------------------------------")
# print('1.Community detection', sorted(map(sorted, next_level_communities)))
# print("--------------------------------------------------------------------------------------------------------------")
# print('2.Modularity 1 ', nx.community.greedy_modularity_communities(G))
# print("--------------------------------------------------------------------------------------------------------------")
# print('3.Modularity 2', nx.community.naive_greedy_modularity_communities(G))
# print("--------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------")
# print('4 Measuring partitions', nx.community.modularity(
#     G, nx.community.label_propagation_communities(G)))
# print("--------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------")
# print('5.girvan_newman', tuple(sorted(c) for c in next(communities_generator)))
# print("--------------------------------------------------------------------------------------------------------------")
