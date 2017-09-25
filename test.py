from utils.mnist_reader import load_mnist
from keras import layers
from keras import models
from keras.models import load_model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
from KaggleChallengeKingsInputHandler import *
from KaggleChallengeKingsDebugging import *
from keras.utils import to_categorical


# building the general model
class WholeNetwork:
    def __init__(self, debug, x_train, train_labels):
        self.debug = debug
        self.x_train = x_train
        self.train_labels = train_labels

    def trainOrLoadAllClasses(self, train=False, num_classes=10):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(Dropout(0.3))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(layers.Dense(num_classes, activation='softmax'))
        if train == True:
            self.model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

            history = self.model.fit(self.x_train, self.train_labels, epochs=15, batch_size=64, validation_split=0.2)
            self.model.save('my_model.h5')
            self.debug.drawLearningSummery(history)
        else:
            self.model = load_model('my_model.h5')

    def runAndEvaluate(self, x_test, y_test):
        y_predict = self.model.predict_classes(x_test, batch_size=64, verbose=1)
        debug.plotConfusionMatrix(y_predict)

class TwoLevelNetwork(WholeNetwork):
    def __init__(self, originLabels, metaLabels):
        self.originLabels = originLabels
        self.metaLabels = metaLabels
        self.meta_y_train = []
        self.meta_y_test = []

    def createMetaLabels(self, mapping):
        self.meta_y_train = np.asarray([mapping[cls] for cls in y_train])
        self.meta_y_test = np.asarray([mapping[cls] for cls in y_test])
        self.meta_y_test = self.meta_y_test.to_categorial(self.meta_y_test)


# loading data from files
x_train, y_train = load_mnist('data/fashion', kind='train')
x_test, y_test = load_mnist('data/fashion', kind='t10k')

x_train = x_train.reshape(60000,28,28,-1)
x_test = x_test.reshape(10000,28,28,-1)

labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

x_train, train_labels, x_test, test_lables = \
    NormalizeInputAndOneHotEncodeOutput(x_train, y_train, x_test, y_test)
debug = KaggleChallengeKingsDebugging(y_test, labels)

# create a vanilla CNN
whole = WholeNetwork(debug)
whole.trainOrLoadAllClasses(x_train, train_labels, train=False)
whole.runAndEvaluate(x_test, y_test)


# create two-level CNN
map_labels = {
    0: 0,
    1: 1,
    2: 0,
    3: 0,
    4: 0,
    5: 2,
    6: 0,
    7: 3,
    8: 4,
    9: 5
    }

meta_labels = {
    0: 'Shirts Family',
    1: 'Trouser',
    2: 'Sandal',
    3: 'Sneaker',
    4: 'Bag',
    5: 'Ankle boot'
    }

twoLevelNetwork = TwoLevelNetwork(originLabels=labels, metaLabels=meta_labels)
twoLevelNetwork.createMetaLabels(mapping=map_labels)