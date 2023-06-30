import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from speed_computation import speed_computation


def preprocess_KMclustering(real_trajectories_path, k, bbox, horizontal_n, vertical_n):
    """
    Preprocess data for K-means clustering.
    """
    # read data
    trajectories = pd.read_csv(real_trajectories_path)
    # change to numpy array
    trajectories = trajectories.values
    # compute speed
    speed_list = []
    for trajectory in trajectories:
        speed_list.append(speed_computation(trajectory, bbox, horizontal_n, vertical_n))
    # convert speed to numpy array
    speed_list = np.array(speed_list)
    # reshape to (1, -1)
    speed_list = speed_list.reshape(-1, 1)
    # apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(speed_list)
    # get cluster labels
    labels = kmeans.labels_
    # for each cluster, get the trajectories and save as csv
    for i in range(k):
        trajectories_in_cluster = trajectories[labels == i]
        trajectories_in_cluster = pd.DataFrame(trajectories_in_cluster)
        trajectories_in_cluster.to_csv("../data/GeoLife/pre-cluster/cluster_" + str(i) + ".csv", index=False)






