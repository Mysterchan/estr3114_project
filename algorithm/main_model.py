from keras.models import Model
import tensorflow as tf
from abc import abstractmethod

CLIP_lower = -0.01
CLIP_upper = 0.01
seed = tf.random.normal([16, 100])


class Original_Gan(Model):
    def __init__(self,gen,dis,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gen = gen
        self.dis = dis

    def compile(self, gen_optimizer,dis_optimizer,gen_loss,dis_loss,**kwargs):
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

    @abstractmethod
    @tf.function
    def training_batch(self,image_loader, k, latent_dimention):
        pass

__all__ = ["Original_Gan", "seed", "CLIP_lower", "CLIP_upper"]