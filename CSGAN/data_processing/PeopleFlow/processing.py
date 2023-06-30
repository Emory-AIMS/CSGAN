import pandas as pd
import numpy as np
import os
import datetime
import warnings
warnings.filterwarnings('ignore')
from coord2grid import coord2grid


def keep_useful_columns(data):
    # keep the first, forth, fifth, sixth, eleventh, and fourteenth columns
    data = data.iloc[:, [0, 3, 4, 5, 10, 13]]
    column_names = ['user_id', 'time', 'lon', 'lat', 'purpose', 'transportation']
    data.columns = column_names
    # concatenate the lon and lat to a tuple, named location
    data['location'] = list(zip(data['lon'], data['lat']))
    lon_min = data['lon'].min()
    lon_max = data['lon'].max()
    lat_min = data['lat'].min()
    lat_max = data['lat'].max()
    # drop the lon and lat columns
    data = data.drop(columns=['lon', 'lat'])
    return data, (lon_min, lon_max, lat_min, lat_max)


def remove_duplicate_users(data):
    data = data.drop_duplicates(subset=['user_id'], keep='first')
    return data


def merge_data(raw_directory, user_id_list):
    big_data = pd.DataFrame()
    folder_list = ['6-7', '7-8', '8-9', '9-10', '10-11', '11-12']
    # set lat_min to inf
    lat_min_o = np.inf
    # set lat_max to -inf
    lat_max_o = -np.inf
    # set lon_min to inf
    lon_min_o = np.inf
    # set lon_max to -inf
    lon_max_o = -np.inf
    # iterate through each directory inside the raw directory
    for directory in folder_list:
        print("---------")
        print(directory)
        print("---------")
        # for each file inside the directory in sorted order
        for file in sorted(os.listdir(raw_directory + directory)):
            # if the file ends with 00.csv, or 15.csv, or 30.csv, or 45.csv
            if file.endswith('00.csv') or file.endswith('15.csv') or file.endswith('30.csv') or file.endswith('45.csv'):
                # read the file
                data = pd.read_csv(raw_directory + directory + '/' + file)
                # keep only the useful columns
                data, (lon_min, lon_max, lat_min, lat_max) = keep_useful_columns(data)
                # remove duplicate users
                data = remove_duplicate_users(data)
                # update lat_min, lat_max, lon_min, lon_max
                lat_min_o = min(lat_min_o, lat_min)
                lat_max_o = max(lat_max_o, lat_max)
                lon_min_o = min(lon_min_o, lon_min)
                lon_max_o = max(lon_max_o, lon_max)
                # keep only the rows that have the user_id in the user_id_list
                data = data[data['user_id'].isin(user_id_list)]
                # concatenate the data to the big_data
                big_data = pd.concat([big_data, data])
    bbox = (lon_min_o, lon_max_o, lat_min_o, lat_max_o)
    # keep only the user_id, time, and location columns
    big_data = big_data[['user_id', 'time', 'location']]
    # change to numpy array
    big_data_user_id = big_data['user_id'].to_numpy()
    big_data_time = big_data['time'].to_numpy()
    # split the time by space and get the second element
    big_data_time = [time.split(' ')[1] for time in big_data_time]
    # change to numpy array
    big_data_time = np.array(big_data_time)
    return_data = pd.DataFrame()
    time_list = ['0600', '0615',
                 '0630', '0645',
                 '0700', '0715',
                 '0730', '0745',
                 '0800', '0815',
                 '0830', '0845',
                 '0900', '0915',
                 '0930', '0945',
                 '1000', '1015',
                 '1030', '1045',
                 '1100', '1115',
                 '1130', '1145',
                 '1200', '1215',
                 '1230', '1245']
    count = 0
    for user_id in user_id_list:
        count += 1
        print("---------")
        print('Count: ' + str(count))
        print('Total: ' + str(len(user_id_list)))
        print(user_id)
        print("---------")
        user_id_row_list = []
        for time in time_list:
            if time == '0600':
                time_stamp = '06:00:00'
            elif time == '0615':
                time_stamp = '06:15:00'
            elif time == '0630':
                time_stamp = '06:30:00'
            elif time == '0645':
                time_stamp = '06:45:00'
            elif time == '0700':
                time_stamp = '07:00:00'
            elif time == '0715':
                time_stamp = '07:15:00'
            elif time == '0730':
                time_stamp = '07:30:00'
            elif time == '0745':
                time_stamp = '07:45:00'
            elif time == '0800':
                time_stamp = '08:00:00'
            elif time == '0815':
                time_stamp = '08:15:00'
            elif time == '0830':
                time_stamp = '08:30:00'
            elif time == '0845':
                time_stamp = '08:45:00'
            elif time == '0900':
                time_stamp = '09:00:00'
            elif time == '0915':
                time_stamp = '09:15:00'
            elif time == '0930':
                time_stamp = '09:30:00'
            elif time == '0945':
                time_stamp = '09:45:00'
            elif time == '1000':
                time_stamp = '10:00:00'
            elif time == '1015':
                time_stamp = '10:15:00'
            elif time == '1030':
                time_stamp = '10:30:00'
            elif time == '1045':
                time_stamp = '10:45:00'
            elif time == '1100':
                time_stamp = '11:00:00'
            elif time == '1115':
                time_stamp = '11:15:00'
            elif time == '1130':
                time_stamp = '11:30:00'
            elif time == '1145':
                time_stamp = '11:45:00'
            elif time == '1200':
                time_stamp = '12:00:00'
            elif time == '1215':
                time_stamp = '12:15:00'
            elif time == '1230':
                time_stamp = '12:30:00'
            elif time == '1245':
                time_stamp = '12:45:00'
            # find the index of the user_id
            user_id_index = np.where(big_data_user_id == user_id)
            # find the index of the time in the time list
            time_index = np.where(big_data_time == time_stamp)
            # find the intersection of the two indices
            intersection_index = np.intersect1d(user_id_index, time_index)
            data = big_data.iloc[intersection_index]
            # if the data is empty, add (999, 999) to the user_id_row_list
            if data.empty:
                grid = 0
                user_id_row_list.append(grid)
            else:
                # add the location to the user_id_row_list
                grid = coord2grid(data.iloc[0]['location'], bbox, 500, 500)
                user_id_row_list.append(grid)
        # add the user_id_row_list as a row to the return_data
        user_id_row_list = np.array(user_id_row_list)
        # if the unique value of the user_id_row_list is less than or equal to 2, skip
        if len(np.unique(user_id_row_list)) <= 2:
            continue
        user_id_row_list = user_id_row_list.reshape(1, -1)
        data_frame_tmp = pd.DataFrame(user_id_row_list)
        return_data = return_data.append(data_frame_tmp)
    return return_data


def main():
    raw_directory = '../../data/PeopleFlow/Raw/'
    first_dataframe_path = '../../data/PeopleFlow/Raw/6-7/08TKY_time_0600.csv'
    first_dataframe = pd.read_csv(first_dataframe_path)
    # keep only the useful columns
    first_dataframe, (lon_min, lon_max, lat_min, lat_max) = keep_useful_columns(first_dataframe)
    # remove duplicate users
    first_dataframe = remove_duplicate_users(first_dataframe)
    # get the user_id_list
    user_id_list = first_dataframe['user_id'].tolist()
    # choose the first 10000 users
    user_id_list = user_id_list[:100000]
    # get the return_data
    return_data = merge_data(raw_directory, user_id_list)
    # save the return_data
    return_data.to_csv('../../data/PeopleFlow/Processed/TKY6-12.csv', index=False)


if __name__ == '__main__':
    main()