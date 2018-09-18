import numpy as np
import tensorflow as tf
import dr_utils

# Prepare data
X_train, Y_train, X_test, Y_test = dr_utils.load_data()
X_train, Y_train, X_test, Y_test = dr_utils.preprocess_data(X_train, Y_train, X_test, Y_test)

# Prepare test image
img = dr_utils.load_img("2_test.png")
img = dr_utils.preprocess_img(img, (28, 28), invert_colors=True)

# Start session
with tf.Session() as sess:
    # Load model
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "saved_model")

    # Check training cost
    J_train = sess.run("J:0", feed_dict={"X:0": X_train, "Y:0": Y_train})
    print("J_train: " + str(J_train))

    # Check accuracy
    Y_pred = sess.run("Y_hat/Softmax:0", feed_dict={"X:0": X_train, "Y:0": Y_train})
    accuracy_train = dr_utils.compute_accuracy(Y_pred, Y_train)
    Y_pred = sess.run("Y_hat/Softmax:0", feed_dict={"X:0": X_test, "Y:0": Y_test})
    accuracy_test = dr_utils.compute_accuracy(Y_pred, Y_test)
    print("accuracy_train: " + str(accuracy_train))
    print("accuracy_test: " + str(accuracy_test))

    # Predict digit for test image
    y_pred = sess.run("Y_hat/Softmax:0", feed_dict={"X:0": img, "Y:0": Y_train})[0]
    y_pred = np.argmax(y_pred)
    print("Predicted digit: " + str(y_pred))
