import argparse
import generator
import discriminator
import helper
from tqdm import tqdm
from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from KMeans_clustering import KMeans_clustering
from KModes_clustering import KModes_clustering
import pandas as pd
import time

import gc
gc.collect()
torch.cuda.empty_cache()

## -----------------------
# for GeoLife
# total_locations = 50626 

# for PeopleFlow
total_locations = 250001
## -----------------------
traj_len = 56
origin = 'zero'
batch_size = 64
gen_embedding_dim = 32
gen_hidden_dim = 32
dis_embedding_dim = 64
dis_hidden_dim = 64
## -----------------------
# # for GeoLife (08-12)
# number_clusters = 6
# num_clusters = 6

# for GeoLife
# number_clusters = 4
# num_clusters = 4

# for Peopleflow (transportation)
number_clusters = 8
num_clusters = 8
## -----------------------
# for GeoLife (08-12)
# trajectories_sampled_per_epoch = 6500

# for GeoLife
# trajectories_sampled_per_epoch = 2500

# for PeopleFlow
trajectories_sampled_per_epoch = 2500
## -----------------------
generator_bidirectional = False
discriminator_bidirectional = True
dis_num_layers = 1
dropout = 0.2
MLE_Train_Epochs = 150
Discriminator_pretrain_steps = 75
Discriminator_train_steps = 5
ADV_train_epochs = 75


def train_generator_MLE(generator, generator_optimizer, oracle, real_samples, num_epochs, device):
    """
    Maximum Likelihood Pretraining for the generator
    """
    all_oracle_loss = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        prev_state_dict = generator.state_dict()
        for i in range(0, trajectories_sampled_per_epoch, batch_size):
            # gen_target is selected from real_samples
            gen_inp, gen_target = helper.prepare_generator_batch(real_samples[i:i + batch_size], origin, device)
            generator_optimizer.zero_grad()
            loss = generator.batchNLLLoss(gen_inp, gen_target)
            loss.backward()
            generator_optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / ceil(trajectories_sampled_per_epoch / float(batch_size)) / traj_len

    #     oracle_loss = helper.batchwise_oracle_nll(generator, oracle, trajectories_sampled_per_epoch, batch_size,
    #                                               traj_len, origin, device)
    #     all_oracle_loss.append(oracle_loss)
    #     if epoch % 5 == 0 or epoch == num_epochs - 1:
    #         print('average_train_NLL = %.4f, oracle_NLL = %.4f' % (total_loss, oracle_loss))
    #     oracle.load_state_dict(prev_state_dict)
    # return all_oracle_loss


def train_generator_PG(generator, generator_optimizer, oracle, discriminator, device):
    """
    The generator is trained using policy gradients using the reward from the discriminator.
    """
    # use the generator to generate trajectories
    generated_samples = generator.sample(batch_size*2)
    gen_input, gen_target = helper.prepare_generator_batch(generated_samples, origin, device)
    # prob_matrix: batch_size * (num_clusters + 1)
    prob_matrix = discriminator.rewardGeneration(gen_target)
    # only obtain the last column of prob_matrix, which is the probability of the trajectory being fake
    rewards = prob_matrix[:, -1]
    rewards = torch.ones(rewards.size()).to(device) - rewards
    # add reward to device
    rewards = rewards.to(device)
    # rewards = discriminator.batchClassify(gen_target) -> output between 0 and 1 (batchsize, 1)
    generator_optimizer.zero_grad()
    pg_loss = generator.batchPGLoss(gen_input, gen_target, rewards)
    pg_loss.backward()
    generator_optimizer.step()

    # # generate samples from the generator and compute oracle NLL
    # oracle_loss = helper.batchwise_oracle_nll(generator, oracle, trajectories_sampled_per_epoch, batch_size,
    #                                           traj_len, origin, device)
    # print('oracle_NLL = %.4f' % oracle_loss)
    # return [oracle_loss]


def train_discriminator(KMeans_clustering, discriminator, discriminator_optimizer, real_samples, cluster_ids,
                        generator, oracle, d_steps, device, verbose=True):
    """
    Training the discriminator on real_samples and generated samples from generator.
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # # generate validation set before training
    # real_trajectories_val = oracle.sample(100)
    # gen_trajectories_val = generator.sample(100)
    # # need to get the cluster_ids of the real trajectories for validation
    # cluster_ids_val = []
    # # change real_trajectories_val from a tensor to a numpy array
    # real_trajectories_val = real_trajectories_val.data.cpu().numpy()
    # # get the cluster_ids of the real trajectories
    # for i in range(real_trajectories_val.shape[0]):
    #     cluster_ids_val.append(KMeans_clustering.get_cluster_id(real_trajectories_val[i]))
    # val_input, val_target = helper.prepare_discriminator_data(real_trajectories_val, gen_trajectories_val,
    #                                                           cluster_ids_val, number_clusters, device)
    d_iterator = tqdm(range(d_steps)) if verbose else range(d_steps)
    for d_step in d_iterator:
        # generate "trajectories_sampled_per_epoch" samples from the generator
        generated_samples = helper.batchwise_sample(generator, trajectories_sampled_per_epoch, batch_size)
        # dis_target: 2*trajectories_sampled_per_epoch
        dis_input, dis_target = helper.prepare_discriminator_data(real_samples, generated_samples,
                                                                  cluster_ids, number_clusters, device)
        for epoch in range(3):
            total_loss = 0
            for i in range(0, 2*trajectories_sampled_per_epoch, batch_size):
                dis_input_tmp, dis_target_tmp = dis_input[i:i+batch_size], dis_target[i:i+batch_size]
                discriminator_optimizer.zero_grad()
                # would be the discriminator output for the current batch (batch_size * )
                dis_output_tmp = discriminator.batchClassify(dis_input_tmp)
                criterion = nn.NLLLoss()
                # criterion = nn.CrossEntropyLoss()
                loss = criterion(dis_output_tmp, dis_target_tmp)
                loss.backward()
                discriminator_optimizer.step()
                total_loss += loss.data.item()
            total_loss /= ceil(2*trajectories_sampled_per_epoch / float(batch_size))

            if verbose and (d_step % 5 == 0 or d_step == d_step-1):
                # val_output = discriminator.batchClassify(val_input)
                # print('average_loss = %.4f, val_loss = %.4f' % (
                #     total_loss, criterion(val_output, val_target).data.item()))
                print('average_loss = %.4f' % (total_loss))


# MAIN
if __name__ == '__main__':
    # get the current time out of k times
    current_time = 2

    # for dataset
    # for PeopleFlow
    # save_dir = 'result_peopleflow_tokyo_6_20_15min/time_{}/'.format(current_time)

    # for GeoLife 6-20 15min
    # save_dir = 'result_GeoLife_6_20_15min/time_{}/'.format(current_time)

    # for GeoLife 6-20 15min (08-12)
    # save_dir = 'result_GeoLife_6_20_15min_08_12/time_{}/'.format(current_time)

    # for PeopleFlow trans vector
    save_dir = 'result_peopleflow_tokyo_trans_mode_vector_clustering/time_{}/'.format(current_time)
    os.makedirs(save_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # np.random.seed(0)
    # torch.manual_seed(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    all_oracle_loss = []
    pretrained_oracle_state_dict_path = os.path.join(save_dir, 'pretrained_oracle.pth')
    pretrained_gen_state_dict_path = os.path.join(save_dir, 'pretrained_gen.pth')
    pretrained_dis_state_dict_path = os.path.join(save_dir, 'pretrained_dis.pth')
    trained_oracle_state_dict_path = os.path.join(save_dir, 'trained_oracle.pth')
    trained_gen_state_dict_path = os.path.join(save_dir, 'trained_gen.pth')
    trained_dis_state_dict_path = os.path.join(save_dir, 'trained_dis.pth')

    print('Preparing Training Data ...')
    # GeoLife data
    # oracle_samples = np.genfromtxt("../../data/GeoLife/GeoLife_processed_6_20_2008_2012_re.csv", delimiter=",")

    # GeoLife data
    # oracle_samples = np.genfromtxt("../../data/GeoLife/GeoLife_processed_6_20_2008.csv", delimiter=",")
    
    # PeopleFlow data
    # oracle_samples = np.genfromtxt("../../data/PeopleFlow/Processed/TKY6-20_2008_10000_re.csv", delimiter=",")
    
    # PeopleFlow data trans vector
    oracle_samples = np.genfromtxt("../../data/PeopleFlow/Processed/TKY6-20_2008_10000_w_trans_re.csv", delimiter=",")
    print(oracle_samples[0])
    oracle_samples = torch.Tensor(oracle_samples).long().to(device)
    oracle = generator.Generator(origin, generator_bidirectional, gen_embedding_dim, gen_hidden_dim, total_locations,
                                 traj_len, device).to(device)
    generator = generator.Generator(origin, generator_bidirectional, gen_embedding_dim, gen_hidden_dim, total_locations,
                                 traj_len, device).to(device)
    discriminator = discriminator.Discriminator(num_clusters, total_locations, dis_embedding_dim, dis_hidden_dim,
                                 dis_num_layers, discriminator_bidirectional, traj_len, dropout, device).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-2)
    discriminator_optimizer = optim.Adagrad(discriminator.parameters())

    # # for KMeans
    # print("Start KMeans Clustering ...")
    # # change oracle_samples from a tensor to a numpy array
    # real_trajectories = oracle_samples.data.cpu().numpy()
    # # for GeoLife
    # bbox = (39.85, 40.0, 116.25, 116.5)
    # horizontal_n = 225
    # vertical_n = 225
    # time_res = 0.25
    # # for PeopleFlow
    # bbox = (138.85, 140.90, 34.95, 36.85)
    # horizontal_n = 500
    # vertical_n = 500
    # time_res = 0.25
    # KM = KMeans_clustering(real_trajectories, num_clusters, bbox, horizontal_n, vertical_n, time_res)
    # labels, centroids = KM.clustering()
    # # get the cluster_ids of the real trajectories
    # cluster_ids = []
    # for i in range(real_trajectories.shape[0]):
    #     cluster_ids.append(KM.get_cluster_id(real_trajectories[i]))

    # for KModes
    print("Start KModes Clustering ...")
    # change oracle_samples from a tensor to a numpy array
    real_trajectories = pd.read_csv("../../data/PeopleFlow/Processed/TKY6-20_2008_10000_w_trans_re.csv")
    corresponding_trans = pd.read_csv("../../data/PeopleFlow/Processed/TKY6-20_2008_10000_transportation_re.csv")
    # for PeopleFlow
    bbox = (138.85, 140.90, 34.95, 36.85)
    horizontal_n = 500
    vertical_n = 500
    time_res = 0.25
    KModes = KModes_clustering(real_trajectories, num_clusters, corresponding_trans)
    labels, centroids, rmv_index = KModes.clustering()
    # get the cluster_ids of the real trajectories
    cluster_ids = labels.tolist()
    real_trajectories = oracle_samples.data.cpu().numpy()
    # for i in range(real_trajectories.shape[0]):
    #     cluster_ids.append(KModes.get_cluster_id(real_trajectories[i]))

    print('Starting Generator MLE Training ...')
    start_time_gen_pre = time.time()
    # all_oracle_loss += train_generator_MLE(generator, generator_optimizer, oracle, oracle_samples, MLE_Train_Epochs,
    #                                        device)
    train_generator_MLE(generator, generator_optimizer, oracle, oracle_samples, MLE_Train_Epochs,
                                           device)
    print("--- %s seconds (Our generator pretraining) ---" % (time.time() - start_time_gen_pre))
    torch.save(oracle.state_dict(), pretrained_oracle_state_dict_path)
    torch.save(generator.state_dict(), pretrained_gen_state_dict_path)
    oracle.load_state_dict(torch.load(pretrained_oracle_state_dict_path))
    generator.load_state_dict(torch.load(pretrained_gen_state_dict_path))

    print('Starting Discriminator Training ...')
    start_time_dis_pre = time.time()
    train_discriminator(KModes, discriminator, discriminator_optimizer, oracle_samples, cluster_ids,
                        generator, oracle, Discriminator_pretrain_steps, device, verbose=False)
    print("--- %s seconds (Our discriminator pretraining) ---" % (time.time() - start_time_dis_pre))
    torch.save(discriminator.state_dict(), pretrained_dis_state_dict_path)
    discriminator.load_state_dict(torch.load(pretrained_dis_state_dict_path))

    print('Starting Adversarial Training ...')
    start_time_adv_train = time.time()
    for epoch in tqdm(range(ADV_train_epochs)):
        prev_state_dict = generator.state_dict()
        # all_oracle_loss += train_generator_PG(generator, generator_optimizer, oracle, discriminator, device)
        train_generator_PG(generator, generator_optimizer, oracle, discriminator, device)
        oracle.load_state_dict(prev_state_dict)
        train_discriminator(KModes, discriminator, discriminator_optimizer, oracle_samples, cluster_ids, generator,
                            oracle, Discriminator_train_steps, device, verbose=False)
    print("--- %s seconds (Our adversarial training) ---" % (time.time() - start_time_adv_train))
    torch.save(oracle.state_dict(), trained_oracle_state_dict_path)
    torch.save(generator.state_dict(), trained_gen_state_dict_path)
    torch.save(discriminator.state_dict(), trained_dis_state_dict_path)

    generated_data_1000 = generator.sample(1000).detach().cpu().numpy()
    with open(os.path.join(save_dir, f'generated_data_1000.npz'), 'wb') as outfile:
        np.savez(outfile, gen_data=generated_data_1000)
    np.savetxt(os.path.join(save_dir, f'generated_data_1000.csv'), generated_data_1000, delimiter=",")

    start_time_inf_2500 = time.time()
    generated_data_2500 = generator.sample(2500).detach().cpu().numpy()
    print("--- %s seconds (Our inference 2500 trajectories) ---" % (time.time() - start_time_inf_2500))
    with open(os.path.join(save_dir, f'generated_data_2500.npz'), 'wb') as outfile:
        np.savez(outfile, gen_data=generated_data_2500)
    np.savetxt(os.path.join(save_dir, f'generated_data_2500.csv'), generated_data_2500, delimiter=",")

    # generate results using the trained model
    generated_data_5000 = generator.sample(5000).detach().cpu().numpy()
    with open(os.path.join(save_dir, f'generated_data_5000.npz'), 'wb') as outfile:
        np.savez(outfile, gen_data=generated_data_5000)
    np.savetxt(os.path.join(save_dir, f'generated_data_5000.csv'), generated_data_5000, delimiter=",")

    # generated_data_10000 = generator.sample(10000).detach().cpu().numpy()
    # with open(os.path.join(save_dir, f'generated_data_10000.npz'), 'wb') as outfile:
    #     np.savez(outfile, gen_data=generated_data_10000)
    # np.savetxt(os.path.join(save_dir, f'generated_data_10000.csv'), generated_data_10000, delimiter=",")

    # # plot NNL
    # plt.plot(range(len(all_oracle_loss)), all_oracle_loss)
    # plt.title('Toy Data')
    # plt.xlabel('Epochs')
    # plt.ylabel('Oracle NLL')
    # plt.savefig(os.path.join(save_dir, f'oracle_nll.png'))

    print("Finish generating syntehtic trajectories.")



