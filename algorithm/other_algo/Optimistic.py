if __name__ == "__main__":
    import sys, os
    print(os.getcwd())  
    sys.path.append(os.getcwd())
    __package__ = "algorithm.other_algo"

from ..main_model import *
import tensorflow as tf
from collections import deque  

class OptimisticGAN(Original_Gan):
    def __init__(self,gen,dis, *args, **kwargs):
        super().__init__(gen,dis,*args,**kwargs)
        self.dis_trial = 0
        self.gan_trial = 0
        self.activate = 0

    @tf.function
    def training_batch(self,real_image, k, latent_dimension, dis_mom = None, gen_mom = None):
        # training logic here
        dis_moment_queue = dis_mom
        gen_moment_queue = gen_mom

        for _ in range(k):
            batch_size = real_image.shape[0]
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

            with tf.GradientTape() as d_tape:
                d_fake = self.dis(fake_image, training=True)
                d_real = self.dis(real_image, training=True)
                d_loss = self.dis_loss(d_real, d_fake)

            d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)
            if dis_moment_queue is None:
                new_d_grad = [(2*g, v) for g, v in zip(d_grad, self.dis.trainable_variables)]
            else:
                new_d_grad = [(2*g - last_g, v) for g, last_g, v in zip(d_grad, dis_moment_queue ,self.dis.trainable_variables)]
            self.dis_optimizer.apply_gradients(new_d_grad)
            dis_moment_queue = d_grad

        with tf.GradientTape() as g_tape:
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))
            fake_score = self.dis(fake_image, training=True)
            g_loss = self.gen_loss(fake_score)

        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)

        if gen_moment_queue is None:
            new_g_grad = [(2*g, v) for g, v in zip(g_grad, self.gen.trainable_variables)]
        else:
            new_g_grad = [(2*g - last_g, v) for g, last_g, v in zip(g_grad, gen_moment_queue, self.gen.trainable_variables)]

        self.gen_optimizer.apply_gradients(new_g_grad)
        gen_moment_queue = g_grad

        self.dis_trial += 1 
        return d_loss, g_loss, dis_moment_queue, gen_moment_queue

    @tf.function
    def training_batch_with_adam(self,real_image, k, latent_dimension, dis_first_mom = None, dis_second_mon = None, gen_first_mom = None, gen_second_mom = None, beta_1 = 0.5, beta_2 = 0.9, epsilon = 1e-8):
        dis_first_moment_queue = dis_first_mom
        dis_second_moment_queue = dis_second_mon
        gen_first_moment_queue = gen_first_mom
        gen_second_moment_queue = gen_second_mom
        for _ in range(k):
            batch_size = real_image.shape[0]
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]))

            with tf.GradientTape() as d_tape:

                d_fake = self.dis(fake_image, training=True)
                d_real = self.dis(real_image, training=True)

                d_loss = self.dis_loss(d_real, d_fake)

            d_grad = d_tape.gradient(d_loss, self.dis.trainable_variables)
            
            if dis_first_moment_queue is None:
                dis_first_moment_queue = d_grad
                dis_second_moment_queue = [tf.square(grad) for grad in d_grad]

                new_d_grad = [(2*g , v) for g, v in zip(d_grad, self.dis.trainable_variables)]

                self.dis_optimizer.apply_gradients(new_d_grad)
            else:
                new_first_momemnt = [(beta_1 * m + (1-beta_1) * grad)/(1 - beta_1 ** self.dis_trial) for m, grad in zip(dis_first_moment_queue, d_grad)]
                new_second_moment = [(beta_2 * m + (1-beta_2) * tf.square(grad))/(1 - beta_2 ** self.dis_trial) for m, grad in zip(dis_second_moment_queue, d_grad)]

                new_d_grad = [(2*m/(tf.sqrt(s) + epsilon) - pm/(tf.sqrt(ps) + epsilon), v) for m, pm, s, ps, v in zip(new_first_momemnt, dis_first_moment_queue, new_second_moment, dis_second_moment_queue, self.dis.trainable_variables)]

                # check if new_d_grad is same direction as d_grad by checking cosine > 0, if not, use d_grad
                # first_new_grad = new_d_grad[-2][0]
                # first_grad = d_grad[-2]
                # check if the new grad is same descent direction with the old grad, 
                # print("=====================================")
                # tf.print("first_new_grad: ",first_new_grad)
                # print("first_new_grad: ",first_new_grad)
                # tf.print("first_grad:", first_grad)
                # print("first_grad:", first_grad)
                # ans =  tf.reduce_sum(tf.reduce_sum(first_new_grad * first_grad))
                # tf.print("Tensor value:", ans)
                # print("=====================================")
                
                self.dis_optimizer.apply_gradients(new_d_grad)

                dis_first_moment_queue = new_first_momemnt
                dis_second_moment_queue = new_second_moment

        with tf.GradientTape() as g_tape: 
            fake_image = self.gen(tf.random.normal([batch_size, latent_dimension]), training=True)
            fake_score = self.dis(fake_image, training=True)
            g_loss = self.gen_loss(fake_score)

        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)

        if gen_first_moment_queue is None:
            gen_first_moment_queue = g_grad
            gen_second_moment_queue = [tf.square(grad) for grad in g_grad]

            new_g_grad = [(2*g, v) for g, v in zip(g_grad, self.gen.trainable_variables)]

            self.gen_optimizer.apply_gradients(new_g_grad)
        
        else:
            new_first_momemnt = [(beta_1 * m + (1-beta_1) * grad)/(1 - beta_1 ** self.gan_trial) for m, grad in zip(gen_first_moment_queue, g_grad)]
            new_second_moment = [(beta_2 * m + (1-beta_2) * tf.square(grad))/(1 - beta_2 ** self.gan_trial) for m, grad in zip(gen_second_moment_queue, g_grad)]

            new_g_grad = [(2*m/(tf.sqrt(s) + epsilon) - pm/(tf.sqrt(ps) + epsilon), v) for m, pm, s, ps, v in zip(new_first_momemnt, gen_first_moment_queue, new_second_moment, gen_second_moment_queue, self.gen.trainable_variables)]
            

            self.gen_optimizer.apply_gradients(new_g_grad)
            # check if new_d_grad is same direction as d_grad by checking cosine > 0, if not, use d_grad
            # first_new_grad = new_g_grad[-2][0]
            # first_grad = g_grad[-2]
            # check if the new grad is same descent direction with the old grad, 
            # print("=====================================")
            # ans = tf.reduce_sum(tf.reduce_sum(first_new_grad * first_grad))
            # tf.print("Tensor value:", ans)
            # print("=====================================")

            gen_first_moment_queue = new_first_momemnt
            gen_second_moment_queue = new_second_moment
        
        self.dis_trial += 1
        self.gan_trial += 1
        return d_loss, g_loss, dis_first_moment_queue, dis_second_moment_queue, gen_first_moment_queue, gen_second_moment_queue


__all__ = ["OptimisticGAN"]