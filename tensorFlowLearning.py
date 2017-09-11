import tensorflow as tf
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
train = pd.read_csv(input_path + 'train.csv', nrows=50000000 if is_on_kaggle else 200000)
test = pd.read_csv(input_path + 'test.csv', nrows=None if is_on_kaggle else 1000000)