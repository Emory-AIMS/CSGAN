import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import radians, cos, sin, asin, sqrt


def coord2grid(coord_tuple, bbox, horizontal_n, vertical_n):
    long_min = bbox[0]
    long_max = bbox[1]
    lat_min = bbox[2]
    lat_max = bbox[3]
    current_lat = coord_tuple[1]
    current_long = coord_tuple[0]
    # check if the current coordinate is within the bounding box
    if current_lat < lat_min or current_lat > lat_max or current_long < long_min or current_long > long_max:
        return 0
    else:
        horizontal_resolution = (long_max - long_min) / horizontal_n
        vertical_resolution = (lat_max - lat_min) / vertical_n
        # coord_tuple is in lon, lat, convert to grid
        x = int((coord_tuple[0] - long_min) / horizontal_resolution)
        y = int((coord_tuple[1] - lat_min) / vertical_resolution)
        grid = y * horizontal_n + x + 1
    return grid


