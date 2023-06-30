import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import torch
from torch.autograd import Variable
from math import ceil
from generator import Generator


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
    # trajectory: a list of coordinates
    for i in range(len(coordinate_list) - 1):
        current_coordinate = coordinate_list[i]
        next_coordinate = coordinate_list[i + 1]
        # if both current and next coordinates are not (999, 999), and current and next coordinates are not the same
        if current_coordinate != (999, 999) and next_coordinate != (999, 999) and current_coordinate != next_coordinate:
            # calculate the haversine distance between current and next coordinates
            distance = haversine(current_coordinate[0], current_coordinate[1], next_coordinate[0], next_coordinate[1])
            speed = distance / time_res
            average_speed += speed
            count += 1
    # check if the count is 0
    if count == 0:
        return 0.0
    else:
        return average_speed / count


def prepare_generator_batch(sampled_trajectories, origin, device):
    """
    Prepare the input and target for the generator.
    Inputs:
        - sampled_trajectories: batch_size * seq_len (Tensor with a sample in each row)
    Returns:
        - gen_input: batch_size * seq_len (same as real_trajectories, but with origin prepended)
        - gen_target: batch_size * seq_len (Variable same as sampled_trajectories)
    """
    batch_size, seq_len = sampled_trajectories.size()
    gen_target = sampled_trajectories
    gen_target = Variable(gen_target).long().to(device)
    gen_input = torch.zeros(batch_size, seq_len)
    if origin == 'zero':
        gen_input[:, 0] = 1
    gen_input[:, 1:] = gen_target[:, :seq_len-1]
    gen_input = Variable(gen_input).long().to(device)
    return gen_input, gen_target


def prepare_discriminator_data(real_trajectories, gen_trajectories, cluster_ids, number_clusters, device):
    """
    Takes real_trajectories, gen_trajectories and prepares dis_input and dis_target for discriminator.
    Inputs:
        - real_trajectories: real_size * seq_len
        - gen_trajectories: gen_size * seq_len
        - cluster_ids: real_size * 1, which stores the cluster id of each real trajectory
    Returns:
        - dis_input: (real_size + gen_size) * seq_len
        - dis_target: real_size + gen_size (boolean 1/0)
    """
    # # change real_trajectories to tensor
    # real_trajectories = torch.from_numpy(real_trajectories).long().to(device)
    # # change gen_trajectories to tensor
    # gen_trajectories = torch.from_numpy(gen_trajectories).long().to(device)
    dis_input = torch.cat((real_trajectories, gen_trajectories), 0).type(torch.LongTensor)
    cluster_ids = np.array(cluster_ids)
    # change cluster_ids to a tensor
    cluster_ids = torch.from_numpy(cluster_ids).type(torch.LongTensor)
    # assign {num_clusters+1} to the generated trajectories
    generated_traj_ids = torch.ones(gen_trajectories.size(0)) * (number_clusters)
    generated_traj_ids = generated_traj_ids.type(torch.LongTensor)
    dis_target = torch.cat((cluster_ids, generated_traj_ids), 0)
    # shuffle
    perm = torch.randperm(dis_target.size()[0])
    dis_input = dis_input[perm]
    dis_target = dis_target[perm]
    dis_input = Variable(dis_input).to(device)
    dis_target = Variable(dis_target).to(device)
    return dis_input, dis_target


def batchwise_sample(generator, num_samples, batch_size):
    generated_trajectories = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        generated_trajectories.append(generator.sample(batch_size))
    return torch.cat(generated_trajectories, 0)[:num_samples]


def batchwise_oracle_nll(generator, oracle, num_samples, batch_size, max_seq_len, origin, device):
    # use the generator to sample trajectories
    sampled_trajectories = batchwise_sample(generator, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        # the only difference between gen_input and gen_target is that the first element is changed to the origin
        gen_input, gen_target = prepare_generator_batch(sampled_trajectories[i:i+batch_size], origin, device)
        oracle_loss = oracle.batchNLLLoss(gen_input, gen_target) / max_seq_len
        oracle_nll += oracle_loss.data.item()
    return oracle_nll/(num_samples/batch_size)



