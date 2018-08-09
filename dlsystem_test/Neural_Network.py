import numpy as np
from Node import *

class nn:
    class SoftMaxOp(Operator):
        """SoftMaxOp is for the softmax function"""
        def __call__(self, input_node, axis=-1):
            exp_input = exp(input_node)
            return exp_input / reduce_sum(exp_input, axis=axis, keepdims=True)

        def compute(self, node):
            assert False, "\033[1;31mSoftMaxNode won't compute\033[0m"

        def gradient(self, node, this_grad):
            assert False, "\033[1;31mSoftMaxNode won't gradient\033[0m"


    class ReluOp(Operator):
        """ReluOp is for the relu function"""
        def __call__(self, input_node):
            new_node = Operator.__call__(self)
            new_node.inputs = [input_node]
            new_node.name = "relu(%s)" % input_node.name
            return new_node

        def compute(self, node):
            node.value = c_boost.relu_boost(node.inputs[0].value)

        def gradient(self, node, this_grad):
            return [relu_gradient(node.inputs[0], this_grad)]


    class SoftMaxCELOp(Operator):
        """SoftMaxCELOp is for the softmax_cross_entropy_with_logits function"""
        def __call__(self, logits = None, labels = None):
            new_node = Operator.__call__(self)
            new_node.inputs = [logits, labels]
            new_node.name = "SoftmaxCrossEntry(%s, %s)" % (logits.name, labels.name)
            return new_node

        def compute(self, node):
            assert len(node.inputs) == 2, "\033[1;31mSoftMaxCrossEntry compute node input args_num is not 2\033[0m"
            tmp_exp = np.exp(node.inputs[0].value)
            tmp_softmax = tmp_exp / np.sum(tmp_exp, axis=-1, keepdims=True)
            node.value = -np.sum(node.inputs[1].value * np.log(tmp_softmax), axis=-1, keepdims=True)

        def gradient(self, node, this_grad):
            return [this_grad * (nn.softmax(node.inputs[0]) - node.inputs[1]), zeros_like(node.inputs[1])]


    softmax = SoftMaxOp()
    relu = ReluOp()
    softmax_cross_entropy_with_logits = SoftMaxCELOp()
    conv2d = Conv2d_Op()
    max_pool = MaxPoolOp()
    dropout = DropOutOp()
