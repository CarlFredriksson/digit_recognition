import math
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img

def preprocess_img(img, size, invert_colors=False):
    if invert_colors:
        img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)

    return img

def load_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    return X_train, Y_train, X_test, Y_test

def visualize_data(X, Y, plot_name):
    plt.subplot(221)
    plt.imshow(X[0], cmap=plt.get_cmap("gray"))
    plt.title("y: " + str(Y[0]))
    plt.subplot(222)
    plt.imshow(X[1], cmap=plt.get_cmap("gray"))
    plt.title("y: " + str(Y[1]))
    plt.subplot(223)
    plt.imshow(X[2], cmap=plt.get_cmap("gray"))
    plt.title("y: " + str(Y[2]))
    plt.subplot(224)
    plt.imshow(X[3], cmap=plt.get_cmap("gray"))
    plt.title("y: " + str(Y[3]))
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def preprocess_data(X_train, Y_train, X_test, Y_test):
    # Normalize image pixel values from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # Change y values from 0-9 to one hot vectors
    Y_train = convert_to_one_hot(Y_train)
    Y_test = convert_to_one_hot(Y_test)

    # Add channels dimension
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, Y_train, X_test, Y_test

def convert_to_one_hot(Y):
    Y_onehot = np.zeros((len(Y), Y.max() + 1))
    Y_onehot[np.arange(len(Y)), Y] = 1

    return Y_onehot

def random_mini_batches(X_train, Y_train, mini_batch_size):
    mini_batches = []
    m = X_train.shape[0] # Number of training examples

    # Shuffle training examples
    permutation = list(np.random.permutation(m))
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train[permutation]

    # Partition into mini-batches
    num_complete_mini_batches = math.floor(m / mini_batch_size)
    for i in range(num_complete_mini_batches):
        X_mini_batch = X_shuffled[i * mini_batch_size : (i + 1) * mini_batch_size]
        Y_mini_batch = Y_shuffled[i * mini_batch_size : (i + 1) * mini_batch_size]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)

    # Handling the case that the last mini-batch < mini_batch_size
    if m % mini_batch_size != 0:
        X_mini_batch = X_shuffled[num_complete_mini_batches * mini_batch_size : m]
        Y_mini_batch = Y_shuffled[num_complete_mini_batches * mini_batch_size : m]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)

    return mini_batches

def compute_accuracy(Y_pred, Y_real):
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_real = np.argmax(Y_real, axis=1)
    num_correct = np.sum(Y_pred == Y_real)
    accuracy = num_correct / Y_real.shape[0]

    return accuracy

def compute_cost(Y, Y_hat):
    # Add small value epsilon to tf.log() calls to avoid taking the log of 0
    epsilon = 1e-10
    J = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_hat + epsilon), axis=1), name="J")

    return J

def create_model(height, width, channels, num_classes):
    tf.reset_default_graph()

    X = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channels), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name="Y")
    training_flag = tf.placeholder_with_default(False, shape=())

    conv1 = tf.layers.conv2d(X, filters=32, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding="valid")
    # Dropout does not apply by default, training=True is needed to make the layer do anything
    # We only want dropout applied during training
    dropout = tf.layers.dropout(pool1, rate=0.2, training=training_flag) 
    flatten = tf.layers.flatten(dropout)
    dense1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
    Y_hat = tf.layers.dense(dense1, num_classes, activation=tf.nn.softmax, name="Y_hat")

    # Compute cost
    J = compute_cost(Y, Y_hat)

    return X, Y, training_flag, Y_hat, J

def run_model(X, Y, training_flag, Y_hat, J, X_train, Y_train, X_test, Y_test, mini_batches, LEARNING_RATE, NUM_EPOCHS):
    # Create train op
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(J)

    # Start session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for epoch in range(NUM_EPOCHS):
            for (X_mini_batch, Y_mini_batch) in mini_batches:
                _, J_train = sess.run([train_op, J], feed_dict={X: X_mini_batch, Y: Y_mini_batch, training_flag: True})
            print("epoch: " + str(epoch) + ", J_train: " + str(J_train))

        # Final costs
        J_train = sess.run(J, feed_dict={X: X_train, Y: Y_train})
        J_test = sess.run(J, feed_dict={X: X_test, Y: Y_test})

        # Compute training accuracy
        Y_pred = sess.run(Y_hat, feed_dict={X: X_train, Y: Y_train})
        accuracy_train = compute_accuracy(Y_pred, Y_train)

        # Compute test accuracy
        Y_pred = sess.run(Y_hat, feed_dict={X: X_test, Y: Y_test})
        accuracy_test = compute_accuracy(Y_pred, Y_test)

        # Save model
        tf.saved_model.simple_save(sess, "saved_model", inputs={"X": X, "Y": Y}, outputs={"Y_hat": Y_hat})

    return J_train, J_test, accuracy_train, accuracy_test
