from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.initializers import HeNormal
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

class DCGAN_Generator(Model):
    def __init__(self):
        super().__init__()
        
        self.layers_ = []
        self.layers_.append(Dense(units=(8*8*512),input_dim=100,activation='relu'))
        self.layers_.append(Reshape((8,8,512)))
        self.layers_.append(Conv2DTranspose(256,4,2,'same',activation='relu',use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(BatchNormalization())
        self.layers_.append(Conv2DTranspose(128,4,2,'same',activation='relu',use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(BatchNormalization())
        self.layers_.append(Conv2DTranspose(3,4,2,'same',activation='tanh', use_bias=False,kernel_initializer=HeNormal()))

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x    
   

    def model(self):
        x = Input(shape=(None, 100))
        return Model(inputs=[x], outputs=self.call(x))

class DCGAN_discriminator(Model):
    def __init__(self, gan_type = "normal"):
        super().__init__()
        self.layers_ =[]
        self.layers_.append(Conv2D(128,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(Conv2D(256,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(BatchNormalization())
        self.layers_.append(Conv2D(512,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(BatchNormalization())
        self.layers_.append(Conv2D(1,4,2,'valid',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=HeNormal()))
        self.layers_.append(Flatten())
        self.layers_.append(Dense(1,kernel_initializer=HeNormal()))
        if gan_type == "normal":
            self.layers_.append(Activation('sigmoid'))
    
    def call(self,x):
        for layer in self.layers_:
            x = layer(x)
        return x
    
    def model(self):
        x = Input(shape=(64,64,3))
        return Model(inputs=x, outputs=self.call(x))

    

__all__ = ["DCGAN_Generator", "DCGAN_discriminator"]

if __name__ == "__main__":
    gen = DCGAN_Generator(64, 3, 2)
    dis = DCGAN_discriminator()
    print(dis(tf.random.normal([32,64,64,3])))
