import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import datetime

import warnings
warnings.filterwarnings('ignore')

import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Conv2DTranspose

from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.backend import set_session


def list_images(basePath, contains=None):

    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)


def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        for filename in filenames:

            if contains is not None and filename.find(contains) == -1:
                continue


            ext = filename[filename.rfind("."):].lower()


            if ext.endswith(validExts):

                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


def load_images(directory='', size=(64, 64)):
    images = []

    imagePaths = list(list_images(directory))

    for path in imagePaths:

        if not ('OSX' in path):
            path = path.replace('\\', '/')

            image = cv2.imread(path)  # Reading the image with OpenCV
            image = cv2.resize(image, size)  # Resizing the image, in case some are not of the same size

            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return images


images=load_images('datas/')


_,ax = plt.subplots(5,5, figsize = (8,8))
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(images[5*i+j])
        ax[i,j].axis('off')


class GAN():
    def __init__(self):
        self.img_shape = (64, 64, 3)

        self.noise_size = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)

        self.discriminator.trainable = False

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined.summary()


    def build_generator(self):
        epsilon = 0.00001
        noise_shape = (self.noise_size,)

        model = Sequential()

        model.add(Dense(4 * 4 * 512, activation='linear', input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 512)))

        model.add(Conv2DTranspose(512, kernel_size=[4, 4], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=[4, 4], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=[4, 4], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, kernel_size=[4, 4], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=[4, 4], strides=[1, 1], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))


        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(128, (3, 3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, metrics_update=50, save_images=100, save_model=2000):

        X_train = np.array(images)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        mean_d_loss = [0, 0]
        mean_g_loss = 0

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)



            d_loss = 0.5 * np.add(self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1))),
                                  self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1))))


            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))

                valid_y = np.array([1] * batch_size)
                g_loss = self.combined.train_on_batch(noise, valid_y)

            mean_d_loss[0] += d_loss[0]
            mean_d_loss[1] += d_loss[1]
            mean_g_loss += g_loss



            if epoch % metrics_update == 0:
                print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (
                epoch, mean_d_loss[0] / metrics_update, 100 * mean_d_loss[1] / metrics_update,
                mean_g_loss / metrics_update))
                mean_d_loss = [0, 0]
                mean_g_loss = 0


            if epoch % save_images == 0:
                self.save_images(epoch)


            if epoch % save_model == 0:
                self.generator.save("generator_%d" % epoch)
                self.discriminator.save("discriminator_%d" % epoch)


    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (10, self.noise_size))
        gen_imgs = self.generator.predict(noise)


        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5, 5, figsize=(8, 8))

        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[5 * i + j])
                axs[i, j].axis('off')


        fig.savefig("Results/T_%d.png" % epoch)
        plt.close()




x = datetime.datetime.now()
print(x)
gan=GAN()
gan.train(epochs=10001, batch_size=256, metrics_update=100, save_images=500, save_model=10000)
x = datetime.datetime.now()
print(x)