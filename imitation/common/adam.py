import numpy as np
import tensorflow as tf
import imitation.common.tf_util as U

class Adam():
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, dtype=tf.float32):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list, dtype=dtype)
        self.getflat = U.GetFlat(var_list)

    def update(self, localg, stepsize):
        localg = localg.astype('float32')

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * localg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (localg * localg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)
