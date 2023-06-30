import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from speed_computation import speed_computation, grid2coord_dictionary, distinct_visit


class KMeans_clustering_derived_feature_vec:
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
        self.inertia = None
        self.grid2coord_dictionary = grid2coord_dictionary(bbox, horizontal_n, vertical_n)
        self.time_res = time_res

    def clustering(self):
        clustering_list = []
        for index in range(len(self.x)):
            trajectory = self.x.iloc[index]
            speed, cumulative_distance = speed_computation(trajectory, self.bbox, self.horizontal_n, self.vertical_n, self.time_res,
                              self.grid2coord_dictionary)
            # compute distinct visit
            num_distinct_visit = distinct_visit(trajectory)
            tmp_list = [speed, cumulative_distance, num_distinct_visit]
            clustering_list.append(tmp_list)
        clustering_list = np.array(clustering_list)
        clustering_list = clustering_list.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(clustering_list)
        labels = kmeans.labels_
        # # add 1 to labels to make it start from 1
        # labels = labels + 1
        self.labels = labels
        centroids = kmeans.cluster_centers_
        self.centroids = centroids
        self.inertia = kmeans.inertia_
        return labels, centroids, kmeans.inertia_

    def get_cluster_id(self, trajectory):
        # trajectory is one of the trajectories in x, find the row index of trajectory in x
        row_index = np.where(self.x == trajectory)[0][0]
        return self.labels[row_index]