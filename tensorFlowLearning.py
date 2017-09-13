import tensorflow as tf
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
train = pd.read_csv(input_path + 'train.csv', nrows=50000000 if is_on_kaggle else 200000)
test = pd.read_csv(input_path + 'test.csv', nrows=None if is_on_kaggle else 1000000)

response = pd.DataFrame()
response['is_attributed'] = train['is_attributed']
train.drop('is_attributed', axis=1, inplace=True)

train.drop(['attributed_time'], axis=1, inplace=True)

test_prediction = pd.DataFrame()
test_prediction['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)