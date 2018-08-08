from Node import *
from assistance import *
import numpy as np


class Session:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def run(obj_nodes, feed_dict = {}):
        """
        :param obj_nodes: nodes whose values need to be computed.
        :param feed_dict: list of variable nodes whose values are supplied by user.
        :return: A list of values for nodes in eval_node_list.
        """
        eval_node_list = obj_nodes if isinstance(obj_nodes, list) else [obj_nodes]
        feed_placeholder(feed_dict)
        topo_order = find_topo_sort(eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceHolderOp) or isinstance(node.op, VariableOp) or isinstance(node.op, MinimizeOp):
                continue
            # print("start", node.op)
            node.op.compute(node)
            # print("finished", node.op)
        return [node.value for node in eval_node_list] if isinstance(obj_nodes, list) else obj_nodes.value


default_session = Session()


def gradients(output_node, node_list):
    """
    :param output_node: out_put node in the current compute graph
    :param node_list: nodes which we are going to cal its gradient
    :return: a new list of graph representing the gradient of the node in the node_list
    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [ones_like(output_node)]
    node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        # print("gradient_started: ", node.name)
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        if node.op.gradient is None:
            continue
        op_grad = node.op.gradient(node, grad)
        for i in range(len(node.inputs)):
            tmp_former_list = node_to_output_grads_list.get(node.inputs[i], [])
            tmp_former_list.append(op_grad[i])
            node_to_output_grads_list[node.inputs[i]] = tmp_former_list
        # print("gradient_finished: ", node.name)
    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


def random_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32, seed=None, name=None):
    return dtype(np.random.normal(loc=mean, scale=stddev, size=shape))


# import dtype from numpy
float16 = np.float16
float32 = np.float32
float64 = np.float64
float128 = np.float128
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

# import train and nn class
from Optimizer import *
from Neural_Network import *




