from assistance import *
import numpy as np
import TensorFreeze as tf
import c_boost
# This file is for the Node and Operation


class Node(object):
    """Node is a basic element in a tenserflow graph"""

    def __init__(self):
        """
            inputs: the fathers(value source) of the current node
            op: refer to the operation of the current node
            value: value of this node:
                * for constant, the value is only the value :)
                * for placeholder, their value is defined from the feed_dict
                * for variables, their value is stored in a var_list, and when
                run the initialize function, their value will be updated
                * for other calculation node, their value is determined in the run
                process, and update when the graph compute is running
            const_value: if constant exists, this store the constant
            name: for debug
        """
        self.inputs = []
        self.op = None
        self.value = None
        self.const_value = None
        self.name = None

    def eval(self, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        return tf.default_session.run(self, feed_dict)

    def run(self, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        return tf.default_session.run(self, feed_dict)

    def __add__(self, other):
        """Add two node will return a new node"""
        if isinstance(other, Node):
            # other is a Node
            new_node = add(self, other)
        else:
            # other is a constant
            new_node = add(self, constant(other))
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub(self, other)
        else:
            new_node = sub(self, constant(other))
        return new_node

    def __rsub__(self, other):
        new_node = sub(constant(other), self)
        return new_node

    def __neg__(self):
        new_node = sub(constant(0), self)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul(self, other)
        else:
            new_node = mul(self, constant(other))
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div(self, other)
        else:
            new_node = div(self, constant(other))
        return new_node

    def __rtruediv__(self, other):
        new_node = div(constant(other), self)
        return new_node

    # Allow the left-hand-side operation
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        return self.name

    __repr__ = __str__


class Operator(object):
    """Operator is the basic class for the operation of the node"""

    def __call__(self):
        """ create a new node with specific operation """
        new_node = Node()
        new_node.op = self
        return new_node

    # Given the value of the input nodes, compute can give the value of the result
    # virtual function
    def compute(self, node):
        """
        :param node: node that is going to be computed
        :return: the value of the node
        """
        raise NotImplementedError

    # build a gradient tree to calculate the grad of node
    # the current node's grad is given in the output_grad
    # virtual function
    def gradient(self, node, this_grad):
        """
        :param node: the grad of this node will be calculated
        :param this_grad: the current node's grad
        :return: a list: the gradient contribution to the input_node of this node,
                the order is same as the order in the inputs of the node
        """
        raise NotImplementedError


class ConstantOp(Operator):
    # Node for the constant value
    def __call__(self, const_value, shape = None, dtype=np.float32):
        new_node = Operator.__call__(self)
        new_node.inputs = []
        new_node.shape = shape
        new_node.dtype = dtype
        if shape is None:
            new_node.const_value = const_value
        else:
            new_node.const_value = dtype(np.array(np.broadcast_to(const_value, shape)))
        new_node.name = "%s" % str(const_value)
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 0, "\033[1;31mConstantOp compute args_num is not 0\033[0m"
        node.value = node.const_value

    def gradient(self, node, this_grad):
        return constant(0) * this_grad


class AddOp(Operator):
    # Operation that add two node into a new node
    def __call__(self, node_left, node_right):
        """
        :param node_left: the left node in the add operation
        :param node_right: the right node in the add operation
        :return: a new node represent node_A + node_B
        """
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s + %s)" % (node_left.name, node_right.name)
        return new_node

    # the implement of the virtual function in the class Operator
    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mAddOp compute args_num is not 2\033[0m"
        node.value = node.inputs[0].value + node.inputs[1].value

    def gradient(self, node, this_grad):
        return [reduce_reshape(this_grad, node.inputs[0]), reduce_reshape(this_grad, node.inputs[1])]


class SubOp(Operator):
    # Op to sub two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s - %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mSubOp compute args_num is not 2\033[0m"
        node.value = node.inputs[0].value - node.inputs[1].value

    def gradient(self, node, this_grad):
        return [reduce_reshape(this_grad, node.inputs[0]), reduce_reshape(-this_grad, node.inputs[1])]


class MulOp(Operator):
    # Op tp multiply two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s * %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mMulOp compute args_num is not 2\033[0m"
        node.value = node.inputs[0].value * node.inputs[1].value

    def gradient(self, node, this_grad):
        return [reduce_reshape(node.inputs[1] * this_grad, node.inputs[0]), reduce_reshape(node.inputs[0] * this_grad, node.inputs[1])]


class DivOp(Operator):
    # Op to div with two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s / %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mDivOp compute args_num is not 2\033[0m"
        node.value = node.inputs[0].value / node.inputs[1].value

    def gradient(self, node, this_grad):
        return [reduce_reshape(1 / node.inputs[1] * this_grad, node.inputs[0]), reduce_reshape(-node.inputs[0] / (node.inputs[1] * node.inputs[1]) * this_grad, node.inputs[1])]


class MatMulOp(Operator):
    """Op to multiply two matrix nodes"""
    def __call__(self, node_left, node_right, trans_left=False, trans_right=False):
        """
        :param node_left: the left matrix
        :param node_right: the right matrix
        :param trans_left: whether to transpose matrix A
        :param trans_right: whether to transpose matrix B
        :return: return a node that refer to the result
        """
        new_node = Operator.__call__(self)
        new_node.left_mat_trans = trans_left
        new_node.right_mat_trans = trans_right
        new_node.inputs = [node_left, node_right]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_left.name, node_right.name, str(trans_left), str(trans_right))
        return new_node

    def compute(self, node):

        node.value = c_boost.matmul_boost(node.inputs[0].value, node.inputs[1].value, node.left_mat_trans, node.right_mat_trans)

    def gradient(self, node, this_grad):
        """Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
                           if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
                           if Y=A B^T, then dA=dY B, dB=dY^T dA
                           if Y=A^T B^T, then dA=B dY^T, dB=dY^T A
        """
        return [matmul(this_grad, node.inputs[1], False, True ^ node.right_mat_trans), matmul(node.inputs[0], this_grad, True ^ node.left_mat_trans, False)]


class ExpOp(Operator):
    """ExpOp is the operator to calculate the exp function"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "exp(%s)" % node_input.name
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31mExpOp compute args_num is not 1\033[0m"
        node.value = np.exp(node.inputs[0].value)

    def gradient(self, node, this_grad):
        return [this_grad * exp(node.inputs[0])]


class LogOp(Operator):
    """LogOp is the operator to calculate the ln functuon"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "log(%s)" % node_input.name
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31mLogOp compute args_num is not 1\033[0m"
        node.value = np.log(node.inputs[0].value)

    def gradient(self, node, this_grad):
        return [1 / node.inputs[0] * this_grad]


class SqrtOp(Operator):
    """SqrtOp is the operator to calculate the sqrt function"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "sqrt(%s)" % node_input.name
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31mSqrtOp compute args_num is not 1\033[0m"
        node.value = np.sqrt(node.inputs[0].value)

    def gradient(self, node, this_grad):
        return [1 / (2 * sqrt(node.inputs[0])) * this_grad]


class PowerOp(Operator):
    """PowerOp is for the pow function"""
    def __call__(self, node_base, node_pow):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_base, node_pow]
        new_node.name = "pow(%s, %s)" % (node_base.name, node_pow.name)
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mPowOp compute args_num is not 2\033[0m"
        node.value = np.power(node.inputs[0].value, node.inputs[1].value)

    def gradient(self, node, this_grad):
        return [node.inputs[1] * pow_op(node.inputs[0], node.inputs[1] - 1) * this_grad, pow_op(node.inputs[0], node.inputs[1]) * log(node.inputs[0]) * this_grad]


class ReduceSumOp(Operator):
    """ReduceSumOp is for the reduce_sum function"""
    def __call__(self, node_input, axis = None, keepdims = False, reduction_indices = None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.keepdims = keepdims
        new_node.name = "reduce_sum(%s)" % node_input.name
        if axis is None and reduction_indices is not None:
            new_node.axis = reduction_indices[0]
        else:
            new_node.axis = axis
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31mReduceSumOp compute args_num is not 1\033[0m"
        node.value = np.sum(node.inputs[0].value, axis=node.axis, keepdims=node.keepdims)

    def gradient(self, node, this_grad):
        return [reduce_sum_gradient(node, this_grad = this_grad)]


class ReduceSumGradientOp(Operator):
    """This is for the gradient of reduce_sum"""
    def __call__(self, node_input, this_grad):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input, this_grad]
        new_node.name = "reduce_sum_grad(%s)" % this_grad.name
        return new_node

    def compute(self, node):
        origin_origin_shape = np.shape(node.inputs[0].inputs[0].value)
        if node.inputs[0].axis is None:
            node.value =  node.inputs[1].value
            node.value = np.array(np.broadcast_to(node.value, origin_origin_shape))
            return
        origin_shape = np.shape(node.inputs[0].value)
        if not node.inputs[0].keepdims:
            shape_list = list(origin_shape)
            shape_list.insert(node.inputs[0].axis, 1)
            new_shape = tuple(shape_list)
        else:
            new_shape = origin_shape
        node.value = node.inputs[1].value
        # Warning: resize won't change the shape of the value
        # Warning: np.broadcast_to returns a view, need copy
        node.value = np.reshape(node.value, new_shape)
        node.value = np.array(np.broadcast_to(node.value, origin_origin_shape))
        assert np.shape(node.value) == origin_origin_shape


    def gradient(self, node, this_grad):
        assert False, "ReduceSumGradient shouldn't appear in gradint"


class ReduceMeanOp(Operator):
    """ReduceMeanOp is for the reduce_sum function"""
    def __call__(self, node_input, axis = None, keepdims = False, reduction_indices = None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.keepdims = keepdims
        new_node.name = "reduce_mean(%s)" % node_input.name
        if axis is None and reduction_indices is not None:
            new_node.axis = reduction_indices[0]
        else:
            new_node.axis = axis
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31mReduceMeanOp compute args_num is not 1\033[0m"
        node.value = np.mean(node.inputs[0].value, axis=node.axis, keepdims=node.keepdims)

    def gradient(self, node, this_grad):
        return [reduce_mean_gradient(node, this_grad = this_grad)]


class ReduceMeanGradientOp(Operator):
    """This is for the gradient of reduce_mean"""
    def __call__(self, node_input, this_grad):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input, this_grad]
        new_node.name = "reduce_mean_grad(%s)" % this_grad.name
        return new_node

    def compute(self, node):
        origin_shape = np.shape(node.inputs[0].value)
        origin_origin_shape = np.shape(node.inputs[0].inputs[0].value)
        if node.inputs[0].axis is None:
            shift = 1
            for dim in origin_origin_shape:
                shift *= dim
            node.value =  node.inputs[1].value / shift
            node.value = np.array(np.broadcast_to(node.value, origin_origin_shape))
            return
        shift = origin_origin_shape[node.axis]
        if not node.inputs[0].keepdims:
            shape_list = list(origin_shape)
            shape_list.insert(node.inputs[0].axis, 1)
            new_shape = tuple(shape_list)
        else:
            new_shape = origin_shape
        node.value = node.inputs[1].value / shift
        node.value = np.reshape(node.value, new_shape)
        node.value = np.array(np.broadcast_to(node.value, origin_origin_shape))
        assert np.shape(node.value) == origin_origin_shape

    def gradient(self, node, this_grad):
        assert False, "ReduceMeanGradient shouldn't appear in gradint"


class ReduceReshapeOp(Operator):
    """This is for BP the gradient that are broadcast by numpy"""
    def __call__(self, this_grad, origin_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [this_grad, origin_input]
        new_node.name = this_grad.name
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 2, "\033[1;31mReduceReshapeOp compute args_num is not 2\033[0m"
        value_A = node.inputs[0].value
        value_B = node.inputs[1].value
        while len(np.shape(value_A)) > len(np.shape(value_B)):
            value_A = np.sum(value_A, axis = 0)
        for dim in range(len(np.shape(value_A))):
            if np.shape(value_A)[dim] > np.shape(value_B)[dim]:
                value_A = np.sum(value_A, axis = dim, keepdims = True)
        node.value = value_A

    def gradient(self, node, this_grad):
        return [this_grad, zeros_like(node.inputs[1])]


class ReShapeOp(Operator):
    """ReShapeOp is for the tf.reshape operation"""
    def __call__(self, input_value, shape):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_value]
        new_node.shape = shape
        new_node.name = "reshape(%s, %s)" % (input_value.name, str(shape))
        return new_node

    def compute(self, node):
        node.value = np.array(np.reshape(node.inputs[0].value, node.shape))

    def gradient(self, node, this_grad):
        return [reshape_grad(node.inputs[0], this_grad)]


class ReShapeGrad(Operator):
    """ReShapeGrad is for the gradient of the reshape operation"""
    def __call__(self, input_value, this_grad):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_value, this_grad]
        new_node.name = "reshape_grad(%s)" % (this_grad.name)
        return new_node

    def compute(self, node):
        node.value = np.reshape(node.inputs[1].value, np.shape(node.inputs[0].value))

    def gradient(self, node, this_grad):
        return [zeros_like(node.inputs[0]), reshape_grad(node.inputs[1], this_grad)]

class EqualOp(Operator):
    """EqualOp is for the equal function"""
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "equal(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node):
        node.value = node.inputs[0].value == node.inputs[1].value

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mEqualOp shouldn't appear in gradient\033[0m"


class CastOp(Operator):
    """CastOp is for the cast function"""
    def __init__(self):
        self.dtype_map = {"float": np.float32, "float32": np.float32, "float64": np.float64, "int": np.int32, "int8": np.int8, "int16": np.int16}
    def __call__(self, cast_input, dtype):
        new_node = Operator.__call__(self)
        new_node.inputs = [cast_input]
        new_node.dtype = self.dtype_map[dtype] if isinstance(dtype, str) else dtype
        if isinstance(cast_input, Node):
            new_node.name = "cast(%s)" % cast_input.name
        else:
            new_node.name = "cast"
        return new_node

    def compute(self, node):
        if isinstance(node.inputs[0], Node):
            node.value = node.dtype(node.inputs[0].value)
        else:
            node.value = node.dtype(node.inputs[0])

    def gradient(self, node, this_grad):
        return [this_grad]


class ArgMaxOp(Operator):
    """ArgMaxOp is for the argmax function"""
    def __call__(self, input_node, axis = None, dimension = None, output_type = np.int64):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_node]
        new_node.axis = axis
        new_node.dimension = dimension
        new_node.output_type = output_type
        new_node.name = "Argmax(%s)" % input_node.name
        return new_node

    def compute(self, node):
        node.value = node.output_type(np.argmax(node.inputs[0].value, axis = node.axis))

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mArgmaxOp shouldn't appear in gradient\033[0m"


class ReluGradientOp(Operator):
    """ReluGradientOp is for the gradient of the relu function"""
    def __call__(self, relu_output, this_grad):
        new_node = Operator.__call__(self)
        new_node.inputs = [relu_output, this_grad]
        new_node.name = "relu_grad(%s, %s)" % (relu_output, this_grad)
        return new_node

    def compute(self, node):
        node.value = c_boost.relu_grad_boost(node.inputs[0].value) * node.inputs[1].value

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mReluGradientOp shouldn't appear in gradient\033[0m"


class Conv2d_Op(Operator):
    """Conv2d_Op is for the conv2d operation"""
    def __call__(self, input, filter, strides, padding):
        new_node = Operator.__call__(self)
        new_node.inputs = [input, filter]
        new_node.name = "conv2d(%s, %s)" % (input.name, filter.name)
        new_node.strides = strides
        new_node.padding = padding
        new_node.width_padding_left = 0
        new_node.width_padding_right = 0
        new_node.height_padding_up = 0
        new_node.height_padding_down = 0
        return new_node

    def compute(self, node):
        """
        Note:
            input_shape: [batch, in_height, in_width, in_channels]
            filter_shape: [filter_height, filter_width, in_channels, out_channels]
        """
        input_tensor = node.inputs[0].value
        filter = node.inputs[1].value
        node.width_padding_left = 0
        node.width_padding_right = 0
        node.height_padding_up = 0
        node.height_padding_down = 0
        if node.padding == 'SAME':
            input_tensor_shape = np.shape(input_tensor)
            filter_shape = np.shape(filter)
            height_padding = filter_shape[0] - 1
            width_padding = filter_shape[1] - 1
            node.width_padding_left = width_padding // 2
            node.width_padding_right = width_padding - node.width_padding_left
            node.height_padding_up = height_padding // 2
            node.height_padding_down = height_padding - node.height_padding_up
        node.value = c_boost.conv2d_boost(input_tensor=input_tensor,
                                          filter=filter,
                                          strides=node.strides,
                                          padding=node.padding,
                                          width_padding_left=node.width_padding_left,
                                          width_padding_right=node.width_padding_right,
                                          height_padding_up=node.height_padding_up,
                                          height_padding_down=node.height_padding_down)

    def gradient(self, node, this_grad):
        return [conv2d_gi(node.inputs[1], this_grad, node),
                conv2d_gw(node.inputs[0], this_grad, node)]


class Conv2d_Input_GradOp(Operator):
    """Conv2d_Input_GradOp is for the conv2d gradient of input_tensor"""
    def __call__(self, filter, this_grad, node):
        new_node = Operator.__call__(self)
        new_node.inputs = [filter, this_grad]
        new_node.name = "conv2d_gi(%s)" % (this_grad.name)
        new_node.origin_node = node
        return new_node

    def compute(self, node):
        filter = node.inputs[0].value
        sensitivity_map = node.inputs[1].value
        # Caution: the padding size has been mirrored
        node.value = c_boost.conv2d_gi_boost(sensitivity_map, filter, node.origin_node.strides, node.origin_node.padding, node.origin_node.width_padding_right, node.origin_node.width_padding_left,
                                             node.origin_node.height_padding_down, node.origin_node.height_padding_up)

    def gradient(self, node, this_grad):
        assert False, "conv2d input grad node shouldn't calculate gradient"


class Conv2d_Filter_GradOp(Operator):
    """Conv2d_Filter_GradOp is for the conv2d gradient of filter"""
    def __call__(self, input_tensor, this_grad, node):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_tensor, this_grad]
        new_node.name = "conv2d_gw(%s)" % (this_grad.name)
        new_node.origin_node = node
        return new_node

    def compute(self, node):
        input_tensor = node.inputs[0].value
        sensitivity_map = node.inputs[1].value
        node.value = c_boost.conv2d_gw_boost(input_tensor, sensitivity_map, node.origin_node.strides, node.origin_node.padding, node.origin_node.width_padding_left,
                                             node.origin_node.width_padding_right, node.origin_node.height_padding_up, node.origin_node.height_padding_down)

    def gradient(self, node, this_grad):
        assert False, "conv2d filter grad node shouldn't calculate gradient"


class MaxPoolOp(Operator):
    """MaxPoolOp is for the maxpool function"""
    def __call__(self, input_tensor, ksize, strides, padding):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_tensor]
        new_node.k_size = ksize
        new_node.strides = strides
        new_node.padding = padding
        new_node.name = "max_pool(%s)" % (input_tensor.name)
        return new_node

    def compute(self, node):
        input_tenser = node.inputs[0].value
        node.width_padding_left = 0
        node.width_padding_right = 0
        node.height_padding_up = 0
        node.height_padding_down = 0
        node.value = c_boost.max_pool_boost(input_tenser, node.k_size, node.strides, node.padding)

    def gradient(self, node, this_grad):
        return[maxpool_grad(node.inputs[0], this_grad, node)]


class MaxPoolGradOp(Operator):
    """MaxPoolGradOp is for the grad of max_pool"""
    def __call__(self, input_tensor, this_grad, node):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_tensor, this_grad]
        new_node.name = "max_pool_grad(%s)" % input_tensor.name
        new_node.origin_node = node
        return new_node

    def compute(self, node):
        input_tensor = node.inputs[0].value
        sensitivity_map = node.inputs[1].value
        node.value = c_boost.max_pool_grad_boost(input_tensor, sensitivity_map, node.origin_node.k_size, node.origin_node.strides)

    def gradient(self, node, this_grad):
        assert False, "MaxPoolGradOp can't calculate gradient"


class DropOutOp(Operator):
    """DropOutOp is for the dropout method"""
    def __call__(self, node_input, keep_prob):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input, keep_prob]
        new_node.name = "DropOut(%s)" % (node_input.name)
        return new_node

    def compute(self, node):
        input_shape = np.shape(node.inputs[0].value)
        keep_prob = node.inputs[1].value
        node.prob_mat = (np.float32)((np.random.rand(*input_shape) < keep_prob) / keep_prob)
        node.value = node.inputs[0].value * node.prob_mat

    def gradient(self, node, this_grad):
        return [dropout_grad(node, this_grad), zeros_like(node.inputs[1])]


class DropOutGrad(Operator):
    """DropOutGrad is for the gradient of the dropout function"""
    def __call__(self, node, this_grad):
        new_node = Operator.__call__(self)
        new_node.inputs = [node, this_grad]
        new_node.name = "Dropout_Grad(%s)" % (this_grad.name)
        return new_node

    def compute(self, node):
        node.value = node.inputs[0].prob_mat * node.inputs[1].value

    def gradient(self, node, this_grad):
        assert False, "DropOutGrad shouldn't gradient"


class ZerosTensorOp(Operator):
    """ZerosTensorOp is to get a tensor with the specific shape which elements are all zeros"""
    def __call__(self, shape, dtype=np.float32, name=None):
        return np.zeros(shape, dtype=dtype)

    def compute(self, node):
        assert False, "\033[1;31mZerosTensorOp doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mZerosTensorOp doesn't support gradient\033[0m"


class OnesTensorOp(Operator):
    """OnesTensorOp is to get a tensor with the specific shape which elements are all ones"""
    def __call__(self, shape, dtype=np.float32, name=None):
        return np.ones(shape, dtype=dtype)

    def compute(self, node):
        assert False, "\033[1;31mOnesTensorOp doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mOnesTensorOp doesn't support gradient\033[0m"


class ZerosLikeOp(Operator):
    """ZerosLikeOp is to get a new node of matrix with the same shape while its elements are all zeros"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "ZerosLike(%s)" % node_input.name
        return new_node

    def compute(self, node):
        assert len(node.inputs) == 1, "\033[1;31m ZerosLikeOp compute args_num is not 1\033[0m"
        shape = np.shape(node.inputs[0].value)
        dtype = node.inputs[0].value.dtype
        if shape == ():
            shape = 1
        node.value = np.zeros(shape = shape, dtype= dtype)

    def gradient(self, node, this_grad):
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Operator):
    """OnesLikeOp is to get a new matrix with the same shape while its elements are all ones"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "OnesLike(%s)" % node_input.name
        return new_node

    def compute(self, node):
        shape = np.shape(node.inputs[0].value)
        dtype = node.inputs[0].value.dtype
        if shape == ():
            shape = 1
        node.value = np.ones(shape=shape, dtype= dtype)

    def gradient(self, node, this_grad):
        return [zeros_like(node.inputs[0])]


class PlaceHolderOp(Operator):
    """PlaceHolderOp is to give a input node a position, the value is need feed"""
    def __call__(self, dtype=np.float32, shape=None, name=None):
        new_node = Operator.__call__(self)
        new_node.shape = shape
        new_node.name = name
        new_node.dtype = dtype
        return new_node

    def compute(self, node):
        assert False, "\033[1;31mPlaceholder doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        return None


class VariableOp(Operator):
    """VariableOp is to define a variable, may with a value"""
    def __init__(self):
        # variable_map is the map from variable to its type and value, usually for the initialize function
        # map value is a list, 0 is the type and 1 is the value of this variable
        self.variable_map = {}
        self.trainable_list = []

    def __call__(self, initial_value=None, trainalbe=True, name=None, dtype=None):
        new_node = Operator.__call__(self)
        new_node.name = name
        new_node.dtype = dtype
        if isinstance(initial_value, Node):
            initial_value = initial_value.const_value
        self.variable_map[new_node] = [dtype, initial_value]
        if trainalbe:
            self.trainable_list.append(new_node)
        return new_node

    def compute(self, node):
        node.value = self.variable_map[node][0](self.variable_map[node][1])

    def gradient(self, node, this_grad):
        return None

    def init_variabels(self):
        for key in self.variable_map:
            value = self.variable_map[key]
            if not value[1] is None:
                if isinstance(value[1], list):
                    value[1] = np.array(value[1])
                key.value = value[1] if value[0] is None else value[1].astype(value[0])


class AssignOp(Operator):
    """for the tf.assign operator"""
    def __call__(self, node_assign, obj):
        new_node = Operator.__call__(self)
        if isinstance(obj, Node):
            new_node.inputs = [node_assign, obj]
            new_node.name = "Assign(%s, %s)" % (node_assign.name, obj.name)
        else:
            new_node.inputs = [node_assign]
            new_node.obj_value = obj
            new_node.name = "Assign(%s, %s)" % (node_assign.name, str(obj))
        return new_node

    def compute(self, node):
        if len(node.inputs) == 2:
            node.inputs[0].value = node.inputs[1].value
            node.value = node.inputs[0].value
        else:
            node.inputs[0].value = np.array(node.obj_value) if isinstance(node.obj_value, list) else node.obj_value

    def gradient(self, node, this_grad):
        return None


class GlobalInitializerOp(Operator):
    """for the global_initializer function"""
    def __call__(self):
        new_node = Operator.__call__(self)
        new_node.name = "GlobalInitializer"
        return new_node

    def compute(self, node):
        Variable.init_variabels()

    def gradient(self, node, this_grad):
        assert False, "GlobalInitializer shouldn't appear in gradint"


class MinimizeOp(Operator):
    """MinimizeOp is to assist the Optimizer to calculate the train"""
    def __call__(self, input_nodes):
        new_node = Operator.__call__(self)
        new_node.inputs = input_nodes
        new_node.name = "Optimizer_Minimize(%s)" % (str([node.name for node in input_nodes]))
        return new_node

    def compute(self, node):
        assert False, "\033[1;31mMinimizeOp shouldn't be compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mMinimizeOp shouldn't be gradient\033[0m"

# some object od the class Operation
constant = ConstantOp()
add = AddOp()
sub = SubOp()
mul = MulOp()
div = DivOp()
matmul = MatMulOp()
exp = ExpOp()
log = LogOp()
sqrt = SqrtOp()
pow_op = PowerOp()
reduce_sum = ReduceSumOp()
reduce_sum_gradient = ReduceSumGradientOp()
reduce_mean = ReduceMeanOp()
reduce_mean_gradient = ReduceMeanGradientOp()
reduce_reshape = ReduceReshapeOp()
reshape = ReShapeOp()
reshape_grad = ReShapeGrad()
equal = EqualOp()
argmax = ArgMaxOp()
cast = CastOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
zeros = ZerosTensorOp()
ones = OnesTensorOp()
placeholder = PlaceHolderOp()
Variable = VariableOp()
assign = AssignOp()
global_variables_initializer = GlobalInitializerOp()
minimize_op = MinimizeOp()
relu_gradient = ReluGradientOp()
conv2d_gi = Conv2d_Input_GradOp()
conv2d_gw = Conv2d_Filter_GradOp()
maxpool_grad = MaxPoolGradOp()
dropout_grad = DropOutGrad()
