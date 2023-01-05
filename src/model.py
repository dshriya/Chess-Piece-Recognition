import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

classes = ['bishop', 'pawn', 'knight', 'rook','queen','king']
num_classes = len(classes)


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', input_shape=(300, 300, 3), activation='relu'))
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        plot_model(self.model, to_file='model.png', show_shapes=True)
        self.model.summary()

    def train(self, batch_size=64, epochs=720):
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            'data\\train',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
            'data\\validation',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        fit_generator = self.model.fit_generator(
            train_generator,
            steps_per_epoch=1543 / batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=314 / batch_size)

        self.visualize(epochs, fit_generator)
        self.__save_weights('%s_epochs_model_weights.h5' % epochs)

    @staticmethod
    def visualize(epochs, fit_generator):
        for key in fit_generator.history.keys():
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(np.arange(0, epochs), fit_generator.history[key], label=key)
            plt.title(key)
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend(loc='lower left')
            plt.savefig(os.path.join('plots', '{}_model_{}_plot.png'.format(epochs, key)))

    def predict(self, weights_file, img):
        if self.model.weights is None:
            self.__load_weights(weights_file)
        score = self.model.evaluate()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def __save_weights(self, weights_file):
        self.model.save_weights(os.path.join('weights', weights_file))

    def __load_weights(self, weights_file):
        self.model.load_weights(os.path.join('weights', weights_file))