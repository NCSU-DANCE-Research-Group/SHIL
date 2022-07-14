"""

Auto Encoder for Anomaly Detection
Builds an auto-encoder with TensorFlow to compress application's system call freq vectors to a
lower latent space and then reconstruct them.

2 layers:
- input layer
- hidden layer 1
- hidden layer 2
- output layer
- sigmoid activation


######
Basic AutoEncoder Tutorial Reference

Builds a 2 layer auto-encoder with TensorFlow to compress MNIST dataset's handwritten digit vectors to a
lower latent space and then reconstruct them.

Consists of: input layer, hidden layer 1, hidden layer 2, output layer,
with neurons, all of which use sigmoid activation

References:
    Aymeric Damien

    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.


"""

import pickle
from tensorflow import set_random_seed
import tensorflow as tf
import random as rn
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


rn.seed(1)
np.random.seed(1)
set_random_seed(1)


def test(train_app,  data_test, basedir='./', get_recon_error=False):
    tf.logging.set_verbosity(tf.logging.ERROR)
    # Saver() prep
    model_save_dir = os.path.join(basedir, 'models/{}/'.format(train_app))
    model_name = 'tomcat'

    # standardize data (counts)
    scaler = None
    with open(os.path.join(model_save_dir, "{}.pkl".format(train_app)), 'rb') as fin:
        scaler = pickle.load(fin)

    dataset_test = scaler.transform(data_test.iloc[:, 1:])
    # shape
    rows_test, columns_test = dataset_test.shape

    with open(os.path.join(model_save_dir, "{}.txt".format(train_app))) as fin:
        line = fin.readline().strip()
        anomaly_threshold = float(line)
        #print("Threshold is: {}".format(anomaly_threshold))

    # pred_labels is autoencoder's prediction
    pred_labels = np.zeros(rows_test)
    reconstruction_errors = np.zeros(rows_test)

    ########################
    # AUTOENCODER START
    ########################

    # Testing Parameters
    batch_size_test = 256
    tf.reset_default_graph()
    # Start a new TF session
    with tf.Session() as sess:

        # LOAD/RESTORE TRAINED MODEL
        # restore network
        saver = tf.train.import_meta_graph(model_save_dir+model_name+'.meta')
        # load parameters/variables
        saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))

        # re define variables & operations
        graph = tf.get_default_graph()

        # tf Graph input
        X = graph.get_tensor_by_name('X:0')
        # Re-construct model
        # name gotten from initial training
        decoder_op = graph.get_tensor_by_name('Sigmoid_3:0')

        # Testing
        num_batches_test = int(rows_test / batch_size_test)

        for j in range(num_batches_test+1):
            # Prepare Data in order
            batch_start = j * batch_size_test
            batch_end = (j + 1) * batch_size_test
            # last batch
            if batch_end > rows_test:
                batch_end = rows_test
            if batch_start >= batch_end:
                continue

            batch_x = dataset_test[batch_start:batch_end, :]
            # Encode and decode the batch
            g_pred = sess.run(decoder_op, feed_dict={X: batch_x})
            g_true = batch_x

            # Get loss (for each sample in batch)
            # arg: keepdims=True would keep it a column vector. row/list better for later processing
            loss = sess.run(tf.reduce_mean(tf.pow(g_true - g_pred, 2), 1))

            # Declare anomaly if loss is greater than threshold tf.greater
            batch_labels = tf.cast(tf.greater_equal(
                loss, anomaly_threshold), tf.int64).eval()
            # batch_labels to pred_labels
            pred_labels[batch_start: batch_end] = batch_labels
            reconstruction_errors[batch_start: batch_end] = loss
        sess.close()
    if get_recon_error:
        return pred_labels, reconstruction_errors, anomaly_threshold
    else:
        return pred_labels
