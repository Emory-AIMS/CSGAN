import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import radians, cos, sin, asin, sqrt


def coord2grid(coord_tuple, bbox, horizontal_n, vertical_n):
    if coord_tuple == (999, 999):
        return 0
    else:
        long_min = bbox[0]
        long_max = bbox[1]
        lat_min = bbox[2]
        lat_max = bbox[3]
        horizontal_resolution = (long_max - long_min) / horizontal_n
        vertical_resolution = (lat_max - lat_min) / vertical_n
        # coord_tuple is in lon, lat, convert to grid
        x = int((coord_tuple[0] - long_min) / horizontal_resolution)
        y = int((coord_tuple[1] - lat_min) / vertical_resolution)
        grid = y * horizontal_n + x + 1
    return grid


