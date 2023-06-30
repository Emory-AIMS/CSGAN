import pandas as pd
from kmodes.kmodes import KModes
import numpy as np
import Levenshtein as lev


class KModes_clustering:
    def __init__(self,
                 real_trajectories,
                 num_clusters,
                 corresponding_trans_mode):
        # x: numpy array
        real_trajectories_tmp = np.array(real_trajectories)
        self.x = real_trajectories_tmp
        self.num_clusters = num_clusters
        self.trans_mode = corresponding_trans_mode
        self.trans_mode_updated = None
        self.labels = None
        self.centroids = None
        self.rm_index = None

    def clustering(self):
        transportation = self.trans_mode
        transportation = np.array(transportation)
        transportation_updated = []
        rmv_index = []
        # for each row, remove all 97
        for i in range(137, len(transportation)):
            current_row = transportation[i]
            current_row = current_row[current_row != 97]
            # if current_row is empty, keep the index of the row
            if len(current_row) == 0:
                rmv_index.append(i)
                continue
            # get the distinct transportation modes
            distinct_modes = np.unique(current_row)
            # get the only most frequent transportation mode
            most_frequent_mode = np.bincount(current_row).argmax()
            # if distinct_modes length is 4, then append it to transportation_updated
            if len(distinct_modes) == 4:
                tmp_list = distinct_modes.tolist()
                transportation_updated.append(tmp_list)
            elif len(distinct_modes) < 4:
                # append the most frequent transportation mode to distinct_modes until reaching 4
                distinct_modes = distinct_modes.tolist()
                while len(distinct_modes) < 4:
                    distinct_modes.append(most_frequent_mode)
                transportation_updated.append(distinct_modes)
            else:
                rmv_index.append(i)
                continue
        # transportation_updated is a list of numpy arrays, convert it to numpy array with shape (n, 4)
        shadow = np.zeros((len(transportation_updated), 4))
        for i in range(len(transportation_updated)):
            shadow[i] = transportation_updated[i]
        transportation_updated = shadow
        self.trans_mode_updated = transportation_updated
        # reshape the transportation_updated
        transportation_updated = transportation_updated.reshape(transportation_updated.shape[0], 4)
        kmodes = KModes(n_clusters=self.num_clusters, random_state=0).fit(transportation_updated)
        labels = kmodes.labels_
        self.labels = labels
        centroids = kmodes.cluster_centroids_
        self.centroids = centroids
        self.rm_index = rmv_index
        return labels, centroids, rmv_index

    def get_cluster_id(self, trajectory):
        # trajectory is one of the trajectories in x, find the row index of trajectory in x
        row_index = np.where(self.x == trajectory)[0][0]
        return self.labels[row_index]

    def inertia(self):
        transportation = np.array(self.trans_mode_updated)
        inertia = 0.0
        for i in range(len(self.trans_mode_updated)):
            cluster_id_tmp = self.labels[i]
            centroid_tmp = self.centroids[cluster_id_tmp]
            trans_tmp = transportation[i]
            distance = lev.distance
            inertia += distance(trans_tmp, centroid_tmp)
        return inertia