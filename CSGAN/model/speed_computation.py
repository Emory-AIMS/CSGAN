import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd


def grid2coord_helper(bbox, horizontal_n, vertical_n):
    coordinate_matrix_x = np.zeros((vertical_n, horizontal_n))
    coordinate_matrix_y = np.zeros((vertical_n, horizontal_n))
    long_min = bbox[0]
    long_max = bbox[1]
    lat_min = bbox[2]
    lat_max = bbox[3]
    x_resolution = (long_max - long_min) / horizontal_n
    y_resolution = (lat_max - lat_min) / vertical_n
    # each grid is represented by the center of the grid
    # thus, the center of the first grid is (long_min + x_resolution/2, lat_min + y_resolution/2)
    # the center of the last grid is (long_max - x_resolution/2, lat_max - y_resolution/2)
    for i in range(vertical_n):
        for j in range(horizontal_n):
            # calculate the center of the grid
            x = long_min + (j + 0.5) * x_resolution
            y = lat_min + (i + 0.5) * y_resolution
            # # y should be calculated in the reverse order
            # y = lat_max - (i + 0.5) * y_resolution
            # store the center of the grid in the matrix
            coordinate_matrix_x[i][j] = x
            coordinate_matrix_y[i][j] = y
    return coordinate_matrix_x, coordinate_matrix_y


def grid2coord_dictionary(bbox, horizontal_n, vertical_n):
    total_grids = horizontal_n * vertical_n
    coordinate_matrix_x, coordinate_matrix_y = grid2coord_helper(bbox, horizontal_n, vertical_n)
    coordinate_dictionary = {}
    for grid in range(1, total_grids + 1):
        # find the location of the grid in the matrix
        i = int((grid - 1) / horizontal_n)
        j = int((grid - 1) % horizontal_n)
        # store the center of the grid in the dictionary
        coordinate_dictionary[grid] = (coordinate_matrix_x[i][j], coordinate_matrix_y[i][j])
    return coordinate_dictionary


def grid2coord(trajectory, bbox, horizontal_n, vertical_n, lookup_dict):
    # change trajectory to a list
    trajectory = trajectory.tolist()
    coordinate_list = []
    for grid_id in trajectory:
        if grid_id == 0:
            # (999, 999) is the default value of the grid
            coordinate_list.append((999, 999))
        else:
            coordinate_list.append(lookup_dict[grid_id])
    return coordinate_list


def haversine(lat1, lon1, lat2, lon2):
    # this is in miles.  For Earth radius in kilometers use 6372.8 km
    R = 6372.8
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def speed_computation(trajectory, bbox, horizontal_n, vertical_n, time_res, lookup_dict):
    # change trajectory to a list of coordinates
    coordinate_list = grid2coord(trajectory, bbox, horizontal_n, vertical_n, lookup_dict)
    average_speed = 0.0
    count = 0
    cumulative_distance = 0.0
    # trajectory: a list of coordinates
    for i in range(len(coordinate_list) - 1):
        current_coordinate = coordinate_list[i]
        next_coordinate = coordinate_list[i + 1]
        # if both current and next coordinates are not (999, 999), and current and next coordinates are not the same
        if current_coordinate != (999, 999) and next_coordinate != (999, 999) and current_coordinate != next_coordinate:
            # calculate the haversine distance between current and next coordinates
            distance = haversine(current_coordinate[0], current_coordinate[1], next_coordinate[0], next_coordinate[1])
            cumulative_distance += distance
            speed = distance / time_res
            average_speed += speed
            count += 1
    # check if the count is 0
    if count == 0:
        return 0.0, cumulative_distance
    else:
        return average_speed / count, cumulative_distance


def distinct_visit(trajectory):
    num_distinct_visit = len(np.unique(trajectory))
    return num_distinct_visit