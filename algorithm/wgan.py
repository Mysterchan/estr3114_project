if __name__ == "__main__":
    import sys, os
    print(os.getcwd())  
    sys.path.append(os.getcwd())
    __package__ = "algorithm"


from .main_model import *
import tensorflow as tf


class wgan(Original_Gan):
    def __init__(self,gen,dis, *args, **kwargs):
        super().__init__(gen,dis,*args,**kwargs)
    @tf.function
    def training_batch(self,real_image, k, latent_dimension):
        for _ in range(k):
            batch_size = real_image.shape[0]
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

            with tf.GradientTape() as d_tape:

                d_fake = self.dis(fake_image, training=True)
                d_real = self.dis(real_image, training=True)

                d_loss = self.dis_loss(d_real, d_fake)

            d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)
            self.dis_optimizer.apply_gradients(zip(d_grad, self.dis.trainable_variables))

            for w in self.dis.trainable_variables:
                w.assign(tf.clip_by_value(w, CLIP_lower, CLIP_upper))

        with tf.GradientTape() as g_tape:
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))
            fake_score = self.dis(fake_image, training=True)
            g_loss = self.gen_loss(fake_score)
            g_loss = tf.reduce_mean(g_loss)

        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(g_grad, self.gen.trainable_variables))

        return d_loss, g_loss


all_ = ["wgan"]