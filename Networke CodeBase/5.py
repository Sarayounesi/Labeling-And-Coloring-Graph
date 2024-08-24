from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


nodes = pd.read_excel('nodes.xlsx', usecols=['NodeId', 'Labels'])
edges = pd.read_excel('edges.xlsx', usecols=['sourceNodeId', 'targetNodeId'])

G = nx.from_pandas_edgelist(edges, 'sourceNodeId', 'targetNodeId')

Not_lable = [1135589,
             1135746,
             1135750,
             1135894,
             1135899,
             1135955,
             1136040,
             1136110,
             1136310,
             1136342,
             1136393,
             1136397,
             1136422,
             1136442,
             1136446,
             1136447,
             1136449,
             1136631,
             1136634,
             1136791,
             1136814,
             1137140,
             1137466,
             1138027,
             1138043,
             1138091,
             1138619,
             1138755,
             1138968,
             1138970,
             1139009,
             1139195,
             1139928,
             1140040,
             1140230,
             1140231,
             1140289,
             1140543,
             1140547,
             1140548,
             1152075,
             1152143,
             1152150,
             1152162,
             1152179,
             1152194,
             1152244,
             1152259,
             1152272,
             1152277,
             1152290,
             1152307,
             1152308,
             1152358,
             1152379,
             1152394,
             1152421,
             1152436,
             1152448,
             1152490,
             1152508,
             1152564,
             1152569,
             1152633,
             1152663,
             1152673,
             1152676,
             1152711,
             1152714,
             1152740,
             1152761,
             1152821,
             1152858,
             1152859,
             1152896,
             1152904,
             1152910,
             1152917,
             1152944,
             1152958,
             1152959,
             1152975,
             1152991,
             1153003,
             1153014,
             1153024,
             1153031,
             1153056,
             1153064,
             1153065,
             1153091,
             1153097,
             1153101,
             1153106,
             1153148,
             1153150,
             1153160,
             1153166,
             1153169,
             1153183,
             1153195,
             1153254,
             1153262,
             1153264,
             1153275,
             1153280,
             1153287,
             1153577,
             1153703,
             1153724,
             1153728,
             1153736,
             1153784,
             1153786,
             1153811,
             1153816,
             1153853,
             1153860,
             1153861,
             1153866,
             1153877,
             1153879,
             1153889,
             1153891,
             1153896,
             1153897,
             1153899,
             1153900,
             1153922,
             1153933,
             1153942,
             1153943,
             1153945,
             1153946,
             1154012,
             1154042,
             1154068,
             1154071,
             1154074,
             1154076,
             1154103,
             1154123,
             1154124,
             1154169,
             1154173,
             1154176,
             1154229,
             1154230,
             1154232,
             1154233,
             1154251,
             1154276,
             1154459,
             1154500,
             1154520,
             1154524,
             1154525,
             1155073]

for nodes in G.nodes:
    neighbor_list = [n for n in G.neighbors(nodes)]
    print(f"Neighbor([node])={neighbor_list}")
    print("-------------------------------------------------------------------------------------")

for node in G.nodes:
    print(f"Degree({node}) = {G.degree(node)}")

for node in G.nodes:
    neighbor_list = [n for n in G.neighbors(node)]
    print(f"Neighbor({node})={neighbor_list}")
    print("-------------------------------------------------------------------------------------")

print(list(G.neighbors(35)))

# K Nearest Neighbors with Python
# Import Libraries


# Load the Data
df = pd.read_csv("Classified Data", index_col=0)
df.head()

# Standardize the Variables
# Because the KNN classifier predicts the class of a given test observation
# by identifying the observations that are nearest to it, the scale of the
# variables matters. Any variables that are on a large scale will have a much
# larger effect on the distance between the observations, and hence on the KNN
# classifier, than variables that are on a small scale.

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'],
                                                    test_size=0.30)


# Using KNN
# Remember that we are trying to come up with a model to predict whether someone
# will TARGET CLASS or not. We'll start with k=1.


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                     weights='uniform')


pred = knn.predict(X_test)

# Predicting and evavluations
# Let's evaluate our knn model.


print(confusion_matrix(y_test, pred))
# [[125  18]
# [ 13 144]]


# print(classification_report(y_test,pred))


# Let's go ahead and use the elbow method to pick a good K Value:

error_rate = []


# Will take some time

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))

plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

# Output for this code can be viewed at : https://tinyurl.com/y8p2kddm

# Here we can see that that after arouns K>23 the error rate just tends to hover
# around 0.06-0.05 Let's retrain the model with that and check the classification report!

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# [[125  18]
# [ 13 144]]


# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# [[132  11]
# [  5 152]]


print(nx.single_source_shortest_path_length(G))
print(nx.floyd_warshall_predecessor_and_distance(G))
print(nx.single_source_bellman_ford_path(G))
# columns=['NodeId','Labels']
# df = pd.DataFrame(list(zip()), columns=columns)
# df.to_excel('nodes - lable.xlsx', sheet_name='nodes')
# print(nx.k_nearest_neighbors(G))
print("-------------------------------------------------------------------------------------")
