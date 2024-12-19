import matplotlib.pyplot as plt
from keras.models import Model
import tensorflow as tf
import numpy as np

import matplotlib.gridspec as gridspec
from PIL import Image


SEED = tf.random.normal([8, 100])

def image_show(generator: Model, discriminator: Model, iter_num, path, train_data):
    train_data = np.array(list(train_data)[0][:8])
    predictions = generator(SEED, training=False)
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        current_image = predictions[i, :, :, :]
        current_image = (current_image + 1) * 127.5
        current_image = tf.cast(current_image, tf.uint8)
        plt.imshow(current_image)
        plt.axis('off')

    for i in range(8):
        plt.subplot(4, 4, i+9)
        current_image = train_data[i, :, :, :]
        current_image = (current_image + 1) * 127.5
        current_image = tf.cast(current_image, tf.uint8)
        plt.imshow(current_image)
        plt.axis('off')

    plt.savefig(f'{path}\\{iter_num}.png')


    current_score = discriminator(predictions, training=False)
    print("========================================")
    print(f"Current Performance Score under discriminator: {current_score}  Iteration: {iter_num}")
    return predictions


def image_test(generator: Model, path, name):
    predictions = generator(tf.random.normal([8*8, 100], seed = 200), training=False)
    plt.figure(figsize = (8,8))
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

    for i in range(64):
    # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    
        image = predictions[i, :, :, :]
        image = (image + 1) * 127.5
        image = tf.cast(image, tf.uint8)
        plt.imshow(image)
        plt.axis('off')

    plt.savefig(f'{path}\\{name}.png')
    return predictions

def generate_10000(generator: Model, path):
    for i in range(300):
        predictions = generator(tf.random.normal([32, 100]), training=False)
        for j in range(32):
            image  = predictions[j, :, :, :]
            image = (image + 1) * 127.5
            image = tf.cast(image, tf.uint8)
            # to numpy
            image = image.numpy()
            image = Image.fromarray(image)
            image.save(f'{path}\\{i*32+j}.png')
    return predictions