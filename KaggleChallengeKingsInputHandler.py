from keras.utils import to_categorical


def NormalizeInputAndOneHotEncodeOutput(x_train, y_train, x_test, y_test):
    """Normalize inputs for values between 0-1 and 1 hot encode output labels"""
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)
    return x_train, train_labels, x_test, test_labels

