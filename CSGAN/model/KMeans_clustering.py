import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from helper import speed_computation, grid2coord_dictionary


class KMeans_clustering:
    def __init__(self,
                 real_trajectories,
                 num_clusters,
                 bbox,
                 horizontal_n,
                 vertical_n,
                 time_res):
        # x: numpy array
        self.x = real_trajectories
        self.num_clusters = num_clusters
        self.bbox = bbox
        self.horizontal_n = horizontal_n
        self.vertical_n = vertical_n
        self.labels = None
        self.centroids = None
        self.grid2coord_dictionary = grid2coord_dictionary(bbox, horizontal_n, vertical_n)
        self.time_res = time_res

    def clustering(self):
        speed_list = []
        for trajectory in self.x:
            speed_list.append(speed_computation(trajectory, self.bbox, self.horizontal_n, self.vertical_n, self.time_res,
                              self.grid2coord_dictionary))
        speed_list = np.array(speed_list)
        speed_list = speed_list.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(speed_list)
        labels = kmeans.labels_
        # # add 1 to labels to make it start from 1
        # labels = labels + 1
        self.labels = labels
        centroids = kmeans.cluster_centers_
        self.centroids = centroids
        return labels, centroids

    def get_cluster_id(self, trajectory):
        # trajectory is one of the trajectories in x, find the row index of trajectory in x
        row_index = np.where(self.x == trajectory)[0][0]
        return self.labels[row_index]


