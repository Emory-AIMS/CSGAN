import pandas as pd
import numpy as np
import os
import datetime
import warnings
warnings.filterwarnings('ignore')
from coord2grid import coord2grid


def find_closest(time_list, target_time):
    # change target_time to date64
    target_time = np.datetime64(target_time)
    time_list = np.array(time_list)
    time_diff = np.abs(time_list - target_time)
    # find the index of the minimum time difference
    min_index = np.argmin(np.abs(time_diff))
    return min_index


def keep_useful_columns(data):
    # concatenate the lon and lat to a tuple, named location
    data['location'] = list(zip(data['lon'], data['lat']))
    # drop the lon and lat columns
    data = data.drop(columns=['lon', 'lat', 'alt', 'label'])
    smallest_time = datetime.datetime(2008, 1, 1, 0, 0, 0)
    largest_time = datetime.datetime(2012, 1, 1, 0, 0, 0)
    return data, smallest_time, largest_time


def merge_data(big_data, user_id_list):
    bbox = (116.25, 116.5, 39.85, 40.0)
    # keep only the useful columns
    big_data, smallest_datetime, largest_datetime = keep_useful_columns(big_data)
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
                 '1230', '1245',
                 '1300', '1315',
                 '1330', '1345',
                 '1400', '1415',
                 '1430', '1445',
                 '1500', '1515',
                 '1530', '1545',
                 '1600', '1615',
                 '1630', '1645',
                 '1700', '1715',
                 '1730', '1745',
                 '1800', '1815',
                 '1830', '1845',
                 '1900', '1915',
                 '1930', '1945']
    start_date = smallest_datetime
    end_date = largest_datetime
    big_data_user_id = big_data['user'].to_numpy()
    while start_date < end_date:
        print("---------")
        print(start_date)
        print("---------")
        for user_id in user_id_list:
            print("---------")
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
                elif time == '1300':
                    time_stamp = '13:00:00'
                elif time == '1315':
                    time_stamp = '13:15:00'
                elif time == '1330':
                    time_stamp = '13:30:00'
                elif time == '1345':
                    time_stamp = '13:45:00'
                elif time == '1400':
                    time_stamp = '14:00:00'
                elif time == '1415':
                    time_stamp = '14:15:00'
                elif time == '1430':
                    time_stamp = '14:30:00'
                elif time == '1445':
                    time_stamp = '14:45:00'
                elif time == '1500':
                    time_stamp = '15:00:00'
                elif time == '1515':
                    time_stamp = '15:15:00'
                elif time == '1530':
                    time_stamp = '15:30:00'
                elif time == '1545':
                    time_stamp = '15:45:00'
                elif time == '1600':
                    time_stamp = '16:00:00'
                elif time == '1615':
                    time_stamp = '16:15:00'
                elif time == '1630':
                    time_stamp = '16:30:00'
                elif time == '1645':
                    time_stamp = '16:45:00'
                elif time == '1700':
                    time_stamp = '17:00:00'
                elif time == '1715':
                    time_stamp = '17:15:00'
                elif time == '1730':
                    time_stamp = '17:30:00'
                elif time == '1745':
                    time_stamp = '17:45:00'
                elif time == '1800':
                    time_stamp = '18:00:00'
                elif time == '1815':
                    time_stamp = '18:15:00'
                elif time == '1830':
                    time_stamp = '18:30:00'
                elif time == '1845':
                    time_stamp = '18:45:00'
                elif time == '1900':
                    time_stamp = '19:00:00'
                elif time == '1915':
                    time_stamp = '19:15:00'
                elif time == '1930':
                    time_stamp = '19:30:00'
                elif time == '1945':
                    time_stamp = '19:45:00'
                # concatenate the date and timestamp
                time_stamp_hms = start_date.strftime('%Y-%m-%d') + ' ' + time_stamp
                # convert to datetime
                date_time = datetime.datetime.strptime(time_stamp_hms, '%Y-%m-%d %H:%M:%S')
                # find the index of the user_id
                user_id_index = np.where(big_data_user_id == user_id)
                dataframe_tmp = big_data.iloc[user_id_index]
                time_list_tmp = dataframe_tmp['time'].to_numpy()
                # find the closest time
                closest_time_index = find_closest(time_list_tmp, date_time)
                # get the entire row
                location = dataframe_tmp.iloc[closest_time_index][2]
                # add the location to the user_id_row_list
                grid = coord2grid(location, bbox, 225, 225)
                user_id_row_list.append(grid)
            # add the user_id_row_list as a row to the return_data
            user_id_row_list = np.array(user_id_row_list)
            # if the unique value of the user_id_row_list is less than or equal to 2, skip
            if len(np.unique(user_id_row_list)) <= 2:
                continue
            user_id_row_list = user_id_row_list.reshape(1, -1)
            data_frame_tmp = pd.DataFrame(user_id_row_list)
            return_data = return_data.append(data_frame_tmp)
        start_date = start_date + datetime.timedelta(days=1)
    return return_data


def main():
    df = read_geolife.read_all_users('Data')
    # saves df to 'geolife.pkl'
    df.to_pickle('geolife.pkl')
    # reads 'geolife.pkl' into df
    df = pd.read_pickle('../../data/GeoLife/geolife.pkl')
    # print the head
    print(df.head())
    user_id_list = df['user'].unique()
    return_data = merge_data(df, user_id_list)
    # save to csv
    return_data.to_csv('../../data/GeoLife/GeoLife_processed_6_20_2008_2012.csv', index=False)


if __name__ == '__main__':
    main()

