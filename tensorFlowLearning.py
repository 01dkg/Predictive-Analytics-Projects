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

def timeEncoding(df):
    df['click_time'] = pd.to_datetime(df['click_time'])

    df['hour'] = df['click_time'].dt.hour
    df['weekday'] = df['click_time'].dt.weekday

    df.drop('click_time', axis=1, inplace=True)
    return df


train = timeEncoding(train)
test = timeEncoding(test)

train_rows = np.shape(train)[0]
train = train.append(pd.DataFrame(data = test))
train['ip'] = train.ip.map(train.groupby('ip').size() / len(train))
test = train[train_rows:np.shape(train)[0]]
train = train[0:train_rows]
del train_rows
scaler = StandardScaler()
scaler.fit(train)
train_dm = scaler.transform(train)
del train
gc.collect()

train_dm_non_anomaly = train_dm[np.where(response == 0)[0]]
train_dm_anomaly = train_dm[np.where(response == 1)[0]]
train_dm_non_anomaly, validation_dm_non_anomaly = train_test_split(train_dm_non_anomaly, test_size=0.2)

niter =2200 if is_on_kaggle else 1500
batch_size = 500000 if np.shape(train_dm_non_anomaly)[0] > 1000000 else int(np.shape(train_dm_non_anomaly)[0] / 5)
learning_rate = 0.004

nOf_features = len(train_dm_non_anomaly[0])
nOf_neurons_first_layer = 6 
nOf_neurons_second_layer = 3
We1 = tf.Variable(tf.random_normal([nOf_features, nOf_neurons_first_layer], dtype=tf.float32))
be1 = tf.Variable(tf.zeros([nOf_neurons_first_layer]))

We2 = tf.Variable(tf.random_normal([nOf_neurons_first_layer, nOf_neurons_second_layer], dtype=tf.float32))
be2 = tf.Variable(tf.zeros([nOf_neurons_second_layer]))

Wd1 = tf.Variable(tf.random_normal([nOf_neurons_second_layer, nOf_neurons_first_layer], dtype=tf.float32))
bd1 = tf.Variable(tf.zeros([nOf_neurons_first_layer]))

Wd2 = tf.Variable(tf.random_normal([nOf_neurons_first_layer, nOf_features], dtype=tf.float32))
bd2 = tf.Variable(tf.zeros([nOf_features]))

X = tf.placeholder(dtype=tf.float32, shape=[None, nOf_features])
encoding = tf.nn.tanh(tf.matmul(X, We1) + be1)
encoding = tf.matmul(encoding, We2) + be2
decoding = tf.nn.tanh(tf.matmul(encoding, Wd1) + bd1)
decoded = tf.matmul(decoding, Wd2) + bd2

loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X, decoded))))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

def get_batch(data, i, size):
    return data[range(i*size, (i+1)*size)]


tf.set_random_seed(1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_train = []
loss_valid = []
loss_anomaly = []
for i in range(niter):
    if batch_size > 0:
    	for j in range(np.shape(train_dm_non_anomaly)[0] // batch_size):
            batch_data = get_batch(train_dm_non_anomaly, j, batch_size)
            sess.run(train_step, feed_dict={X: batch_data})
    else:
        sess.run(train_step, feed_dict={X: train_dm_non_anomaly})
    lt = sess.run(loss, feed_dict={X: train_dm_non_anomaly})
    lv = sess.run(loss, feed_dict={X: validation_dm_non_anomaly})
    la = sess.run(loss, feed_dict={X: train_dm_anomaly})
    loss_train.append(lt)
    loss_valid.append(lv)
    loss_anomaly.append(la)
    if (i % 100 == 0):
        print('iteration {0}: loss train = {1:.4f}, loss valid = {2:.4f}, loss anomaly = {3:.4f}'.format(i, lt, lv, la))

#Final test and outputs
plt.figure()
plt.clf()
plt.cla()
plt.semilogy(loss_train)
plt.semilogy(loss_valid)
plt.semilogy(loss_anomaly)
plt.ylabel('loss')
plt.xlabel('niter')
plt.legend(['train (w/o anomaly)', 'valid (w/o anomaly)', 'anomalistic data'], loc='upper right')
plt.gcf().savefig('learning_curves.png') if is_on_kaggle else plt.show()
np.random.seed(1)
sampled_train_dm_non_anomaly = train_dm_non_anomaly[np.random.choice(len(train_dm_non_anomaly), 5 * np.shape(train_dm_anomaly)[0], replace=False)]
non_anomaly_reconstructed = sess.run(decoded, feed_dict={X: sampled_train_dm_non_anomaly})
anomaly_reconstructed = sess.run(decoded, feed_dict={X: train_dm_anomaly})

non_anomaly_reconstructed_loss = np.sqrt(np.mean(np.square(sampled_train_dm_non_anomaly - non_anomaly_reconstructed), axis=1))
anomaly_reconstructed_loss = np.sqrt(np.mean(np.square(train_dm_anomaly - anomaly_reconstructed), axis=1))
del train_dm_non_anomaly, train_dm_anomaly
gc.collect()

test_dm = scaler.transform(test)
del test
gc.collect()

test_reconstructed = sess.run(decoded, feed_dict={X: test_dm})
test_reconstructed_loss = np.sqrt(np.mean(np.square(test_reconstructed - test_dm), axis=1))
def distribute_in_range(data, min, max):
    max_data = np.max(data)
    a = (max - min) / (max_data - np.min(data))
    b = max - a * max_data
    return a * data + b
test_prediction['is_attributed'] = distribute_in_range(test_reconstructed_loss, 0, 1)

test_prediction.to_csv('test_prediction.csv', index=False)
