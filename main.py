import os
import shutil
import numpy as np
import tensorflow as tf
import dr_utils

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 200

results_file = open("output/results.txt", "w")
try:
    shutil.rmtree("saved_model")
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

# Prepare data
X_train, Y_train, X_test, Y_test = dr_utils.load_data()
dr_utils.visualize_data(X_train, Y_train, "data_visual.png")
X_train, Y_train, X_test, Y_test = dr_utils.preprocess_data(X_train, Y_train, X_test, Y_test)
mini_batches = dr_utils.random_mini_batches(X_train, Y_train, BATCH_SIZE)

# Create model
height, width, channels, num_classes = X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1]
X, Y, training_flag, Y_hat, J = dr_utils.create_model(height, width, channels, num_classes)

# Run model
J_train, J_test, accuracy_train, accuracy_test = dr_utils.run_model(
    X, Y, training_flag, Y_hat, J, X_train, Y_train, X_test, Y_test, mini_batches, LEARNING_RATE, NUM_EPOCHS)
print("accuracy_train: " + str(accuracy_train))
print("accuracy_test: " + str(accuracy_test))
results_file.write("J_train: " + str(J_train) + ", J_test: " + str(J_test) + "\n")
results_file.write("accuracy_train: " + str(accuracy_train) + ", accuracy_test: " + str(accuracy_test) + "\n")
