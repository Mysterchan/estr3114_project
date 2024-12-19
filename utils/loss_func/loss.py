from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.initializers import RandomNormal
from keras.losses import BinaryCrossentropy
import tensorflow as tf

cross_entropy = BinaryCrossentropy(from_logits=False)

### for wgan:
def wgan_gen_loss(fake_output): # tv/indicator
    fake_loss = tf.reduce_mean(fake_output)
    return -fake_loss

def wgan_dis_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = -real_loss + fake_loss
    return total_loss


### for normal gan:
def gan_gen_loss_vanilla(fake_output):   # negative automatically (so mim, more negative, more better) from gan paper
    ones = tf.ones_like(fake_output)
    ones_diff = ones - fake_output
    return -cross_entropy(ones, ones_diff)

def gan_dis_loss_standard(real_output, fake_output, epoch = 1):
    ratio = (50-epoch)/50
    noise_ones = -0.05*tf.random.uniform(tf.shape(real_output)) * (ratio)  # add noise?
    noise_zero = 0.05*tf.random.uniform(tf.shape(fake_output)) * (ratio)
    ones = tf.ones_like(real_output)
    zeros = tf.zeros_like(fake_output)
    disy_realfake = tf.concat([ones,zeros],axis=0)
    dis_noise_realfake = tf.concat([noise_ones,noise_zero],axis=0)
    disy_realfake += dis_noise_realfake
    dis_realfake = tf.concat([real_output,fake_output],axis =0)
    return cross_entropy(disy_realfake,dis_realfake)

def gan_gen_loss_standard(fake_output):   # -log(d)
    ones = tf.ones_like(fake_output)
    return cross_entropy(ones, fake_output)

def gan_gen_loss_js(fake_output, epoch = 1):
    zeros = tf.zeros_like(fake_output)
    noise_zero = 0.05*tf.random.uniform(tf.shape(fake_output)) * (1-epoch/50)
    zeros += noise_zero
    return -cross_entropy(zeros, fake_output)

# vanila gan loss: gan_gen_loss_vanilla / gan_gen_loss_standard + gan_dis_loss_standard
# js gan loss for numericals of gan: gan_gen_loss_js/gan_gen_loss_standard  + gan_dis_loss_standard
    

# JS div

__all__ = ['gan_gen_loss_vanilla', 'gan_gen_loss_standard', 'gan_gen_loss_js', 'gan_dis_loss_standard', 'wgan_gen_loss', 'wgan_dis_loss']    


if __name__ == "__main__":
    # print(wgan_dis_loss(tf.constant([0.5, 0.5]), tf.constant([0.5, 0.5])))
    # print(wgan_dis_loss(tf.constant([0.9, 0.9]), tf.constant([0.5, 0.5])))
    # print(wgan_dis_loss(tf.constant([0.9, 0.9]), tf.constant([0.1, 0.1])))
    print(gan_gen_loss_standard(tf.constant([0.1, 0.1])))
    print(gan_gen_loss_standard(tf.constant([0.5, 0.5])))
    print(gan_gen_loss_standard(tf.constant([0.9, 0.9])))
    print(gan_gen_loss_vanilla(tf.constant([0.1, 0.1])))
    print(gan_gen_loss_vanilla(tf.constant([0.5, 0.5])))
    print(gan_gen_loss_vanilla(tf.constant([0.9, 0.9])))