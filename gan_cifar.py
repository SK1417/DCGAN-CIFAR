import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.datasets import cifar10
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
from imutils import build_montages 
import numpy as np 
import argparse 
import cv2 
import os 

class DCGAN:
    @staticmethod 
    def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):
        model = Sequential()
        inputShape = (dim, dim, depth)
        chanDim = -1

        model.add(Dense(input_dim=inputDim, units=outputDim))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dense(dim*dim*depth))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(256, (5,5), strides=(2,2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(channels, (5,5), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Activation('tanh'))
        return model 
    
    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):

        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Conv2D(64, (5,5), padding='same', strides=(2,2), input_shape=inputShape))
        model.add(LeakyReLU(alpha))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (5,5), padding='same', strides=(2,2), input_shape=inputShape))
        model.add(LeakyReLU(alpha))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (5,5), padding='same', strides=(2,2), input_shape=inputShape))
        model.add(LeakyReLU(alpha))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))


        return model


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', default='output')
ap.add_argument('-e', '--epochs', type=int, default=50)
ap.add_argument('-b', '--batch-size', type=int, default=128)
args = vars(ap.parse_args())

NUM_EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
INIT_LR = 2e-4

print('[INFO] loading dataset...')
((trainX, _), (testX, _)) = cifar10.load_data()
trainImages = np.concatenate([trainX, testX])

images = []
for i in range(len(trainImages)):
    img = cv2.resize(trainImages[i], (64,64))
    images.append(img)
del trainImages

trainImages = np.array(images)
del images
print('Reached')
trainImages = (trainImages.astype('float') - 127.5)/127.5

print('[INFO] building generator...')
gen = DCGAN.build_generator(8, 512, channels=3)

print('[INFO] building discriminator..')
disc = DCGAN.build_discriminator(64, 64, 3)
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR/NUM_EPOCHS)
disc.compile(loss='binary_crossentropy', optimizer=discOpt)

print('[INFO] building GAN...')
disc.trainable = False 
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR/NUM_EPOCHS)
gan.compile(loss='binary_crossentropy', optimizer=discOpt)



print('[INFO] starting training...')
benchmarkNoise = np.random.uniform(-1, 1, size=(256,100))

for epoch in range(NUM_EPOCHS):

    print('[INFO] starting epoch {} of {}...'.format(epoch+1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0]/BATCH_SIZE)

    for i in range(batchesPerEpoch):
        p = None 
        imageBatch = trainImages[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        genImages = gen.predict(noise, verbose=0)

        
        x = np.concatenate([imageBatch, genImages])
        y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
        y = np.reshape(y, (-1,))
        (x, y) = shuffle(x, y)

        discLoss = disc.train_on_batch(x, y)

        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        fakeLabels = [1]*BATCH_SIZE
        fakeLabels = np.reshape(fakeLabels, (-1,))
        
        ganLoss = gan.train_on_batch(noise, fakeLabels)

        if i == batchesPerEpoch - 1:
            p = [args['output'], 'epoch_{}_output.png'.format(str(epoch+1).zfill(4))]
        
        else:

            if epoch < 10 and i%25 == 0:
                p = [args['output'], 'epoch_{}_step_{}.png'.format(str(epoch+1).zfill(4), str(i).zfill(5))]
            
            elif epoch >= 10 and i%100 == 0:
                p = [args['output'], 'epoch_{}_step_{}.png'.format(str(epoch+1).zfill(4), str(i).zfill(5))]
        
        if p is not None:

            print('[INFO] Step {}_{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}'.format(
                epoch+1, i, discLoss, ganLoss
            ))

            images = gen.predict(benchmarkNoise)
            images = ((images*127.5)+127.5).astype('uint8')
            #images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (32,32), (16,16))[0]

            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)    
