if __name__ == "__main__":
    import sys, os
    print(os.getcwd())  
    sys.path.append(os.getcwd())
    __package__ = "algorithm.other_algo"

from ..main_model import *
import tensorflow as tf

class ConsensusGAN(Original_Gan):
    def __init__(self,gen,dis, alpha_rate = 0.1, *args, **kwargs):
        if alpha_rate in kwargs:
            kwargs.pop("alpha_rate")
            alpha_rate = kwargs["alpha_rate"]
        super().__init__(gen,dis,*args,**kwargs)
        self.alpha_rate = alpha_rate

    # @tf.function
    # def training_batch(self,real_image, k, latent_dimension):
    #     '''
    #     simGA with consensus optimization
    #     both dis and gen seek for maximum in their unitity func,
    #     '''
    #     batch_size = real_image.shape[0]
    #     fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

    #     with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
    #         d_fake = self.dis(fake_image, training=True)
    #         d_real = self.dis(real_image, training=True)

    #         d_loss = self.dis_loss(d_real, d_fake)
    #         g_loss = self.gen_loss(d_fake)
        
    #     d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)
    #     g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)

    #     # total_var = self.dis.trainable_variables + self.gen.trainable_variables
    #     # total_grad = d_grad + g_grad
    #     with tf.GradientTape() as total_tape:
    #         regularizer = 1/2 * sum([tf.reduce_sum(tf.square(grad)) for grad in d_grad if grad is not None])

    #     with tf.GradientTape() as total_tape:
    #         regularizer += 1/2 * sum([tf.reduce_sum(tf.square(grad)) for grad in g_grad if grad is not None])


    #     total_var = self.dis.trainable_variables + self.gen.trainable_variables
    #     total_grad = d_grad + g_grad
    #     regularizer_grad = total_tape.gradient(regularizer, total_var)

    #     # gradient renew
    #     grad = [(g + self.alpha_rate * r, v) if r is not None else (g, v) for g, r, v in zip(total_grad, regularizer_grad, total_var)]

    #     self.dis_optimizer.apply_gradients(grad[:len(self.dis.trainable_variables)])
    #     self.gen_optimizer.apply_gradients(grad[len(self.dis.trainable_variables):])

    @tf.function
    def training_batch(self, real_image, k, latent_dimension):
        
        for _ in range(k):
            batch_size = real_image.shape[0]
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

            with tf.GradientTape() as d_tape:
                d_fake = self.dis(fake_image, training=True)
                d_real = self.dis(real_image, training=True)

                d_loss = self.dis_loss(d_real, d_fake)

            d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)

            with tf.GradientTape() as d_grad_tape:
                regularizer = 1/2 * sum([tf.reduce_sum(tf.square(grad)) for grad in d_grad if grad is not None ])
            
            regularizer_grad = d_grad_tape.gradient(regularizer, self.dis.trainable_variables)
            d_gradd = [(g + self.alpha_rate * r, v) if r is not None else (g,v) for g,r,v in zip(d_grad, regularizer_grad, self.dis.trainable_variables)]

            self.dis_optimizer.apply_gradients(d_gradd)
        
        with tf.GradientTape() as g_tape:
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))
            fake_score = self.dis(fake_image, training=True) 
            g_loss = self.gen_loss(fake_score)
            g_loss = tf.reduce_mean(g_loss)
        
        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)

        with tf.GradientTape() as g_grad_tape:
            regularizer = 1/2 * sum([tf.reduce_sum(tf.square(grad)) for grad in g_grad])
        
        regularizer_grad = g_grad_tape.gradient(regularizer, self.gen.trainable_variables)
        g_gradd = [(g + self.alpha_rate * r, v) if r is not None else (g,v) for g,r,v in zip(g_grad, regularizer_grad, self.gen.trainable_variables)]

        self.gen_optimizer.apply_gradients(g_gradd)

        return d_loss, g_loss

    @tf.function
    def training_batch_only_dis(self, real_image, k, latent_dimension):
        for _ in range(k):
            batch_size = real_image.shape[0]
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

            with tf.GradientTape() as d_tape:
                d_fake = self.dis(fake_image, training=True)
                d_real = self.dis(real_image, training=True)

                d_loss = self.dis_loss(d_real, d_fake)
            
            d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)

            with tf.GradientTape() as d_grad_tape:
                regularizer = 1/2 * sum([tf.reduce_sum(tf.square(grad)) for grad in d_grad])
            
            regularizer_grad = d_grad_tape.gradient(regularizer, self.dis.trainable_variables)
            d_gradd = [(g + self.alpha_rate * r, v) if r is not None else (g,v) for g,r,v in zip(d_grad, regularizer_grad, self.dis.trainable_variables)]

            self.dis_optimizer.apply_gradients(d_gradd)
        
        with tf.GradientTape() as g_tape:
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))
            fake_score = self.dis(fake_image, training=True)
            g_loss = self.gen_loss(fake_score)
            g_loss = tf.reduce_mean(g_loss)

        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(g_grad, self.gen.trainable_variables))

        return d_loss, g_loss

_all_ = ["ConsensusGAN"]