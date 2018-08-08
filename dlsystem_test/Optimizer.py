import numpy as np
from Node import *
from assistance import *
from TensorFreeze import gradients


class train:
    """This is a class for some Optimizer"""
    class Optimizer(object):
        """The base class for the Optimizer"""
        def __init__(self, learing_rate = 0.1, name = "Optimizer"):
            self.learning_rate = learing_rate
            self.name=name


    class GradientDescentOptimizer(Optimizer):
        """This is for GradientDescentOptimizer"""
        def __init__(self, learning_rate = 0.1, name = "GradientDescentOptimizer"):
            self.learning_rate = learning_rate
            self.name = name

        def minimize(self, loss_node):
            trainable_list = Variable.trainable_list # trainable variables
            updated_list = [] # variables which need to be updated
            helpful_list = find_topo_sort(node_list=[loss_node]) # variables which is contribute to the loss_node
            for node in trainable_list:
                if node in helpful_list:
                    updated_list.append(node)
            updated_gradient_list = gradients(loss_node, updated_list) # gradient list of the updated_list
            training_list = [] # final nodes which are going to compute
            for pos, node in enumerate(updated_list):
                training_list.append(assign(node, node - (updated_gradient_list[pos] * self.learning_rate)))
            return minimize_op(training_list)


    class AdamOptimizer(Optimizer):
        """This is for the AdamOptimizer"""
        def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, name = "Adam"):
            self.learning_rate = constant(learning_rate)
            self.beta1 = constant(beta1)
            self.beta2 = constant(beta2)
            self.epsilon = constant(epsilon)
            self.name = name

        def minimize(self, loss_node):
            trainable_list = Variable.trainable_list # trainable variables
            gradients_list = gradients(loss_node, trainable_list)
            training_list = []
            # initialize
            t = constant(0)
            m = []
            v = []
            for i in range(len(trainable_list)):
                m.append(constant(0))
                v.append(constant(0))
            assign_t = assign(t, t + 1)
            lr_t = self.learning_rate * sqrt((1 - pow_op(self.beta2, assign_t)) / sqrt(1 - pow_op(self.beta1, assign_t)))
            for pos, variable in enumerate(trainable_list):
                now_m = m[pos]
                now_v = v[pos]
                grad = gradients_list[pos]
                next_m = assign(now_m, self.beta1 * now_m + (1 - self.beta1) * grad)
                next_v = assign(now_v, self.beta2 * now_v + (1 - self.beta2) * grad * grad)
                next_variable = variable - lr_t * next_m / (sqrt(next_v) + self.epsilon)
                training_list.append(assign(variable, next_variable))
            return minimize_op(training_list)
