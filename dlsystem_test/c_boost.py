import ctypes
import numpy as np
from Node import *

boost_lib = ctypes.cdll.LoadLibrary("./boost_cc.so")
boost_conv2d = boost_lib.conv2d
boost_conv2d_gi = boost_lib.conv2d_gi
boost_conv2d_gw = boost_lib.conv2d_gw
boost_max_pool = boost_lib.max_pool
boost_max_pool_grad = boost_lib.max_pool_grad
boost_matmul = boost_lib.matmul


def fecth_pointer(array):
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def matmul_boost(mat_left, mat_right, trans_left, trans_right):
    left_shape = np.shape(mat_left)
    right_shape = np.shape(mat_right)
    m = left_shape[0]
    k = left_shape[1]
    n = right_shape[1]

    if trans_left:
        m = left_shape[1]; k = left_shape[0]
    else:
        m = left_shape[0]; k = left_shape[1]
    if trans_right:
        n = right_shape[0]
    else:
        n = right_shape[1]

    result_mat = np.ndarray((m, n), dtype=np.float32)

    assert not boost_matmul(fecth_pointer(mat_left), fecth_pointer(mat_right), fecth_pointer(result_mat),
                        trans_left, trans_right, m, k, n)
    return result_mat

def conv2d_boost(input_tensor, filter, strides, padding, width_padding_left = 0, width_padding_right = 0,
                 height_padding_up = 0, height_padding_down = 0):
    input_shape = np.shape(input_tensor)  # input x
    filter_shape = np.shape(filter)  # filter w
    assert input_shape[3] == filter_shape[2], "input_tensor and filter in_channel is not equal"
    strides_batch = strides[0]  # strides of batch
    strides_in_height = strides[1]  # strides of height
    strides_in_width = strides[2]  # strides of width
    strides_in_channel = strides[3]  # strides of channel

    if padding == 'SAME':
        input_padded = np.zeros([input_shape[0], input_shape[1] + height_padding_up + height_padding_down,
                                 input_shape[2] + width_padding_left + width_padding_right, input_shape[3]], dtype=np.float32)
        input_padded[:, height_padding_up:input_shape[1] + height_padding_up,
                    width_padding_left: input_shape[2] + width_padding_left, :] = input_tensor
        output_shape = (input_shape[0], input_shape[1], input_shape[2], filter_shape[3])
        output_tensor = np.ndarray(output_shape, dtype=np.float32)
        input_shape = np.shape(input_padded)
    elif padding == 'VALID':
        input_padded = input_tensor
        output_tensor = np.ndarray([input_shape[0], input_shape[1] - filter_shape[0] + 1,
                                  input_shape[2] - filter_shape[1] + 1, filter_shape[3]], dtype=np.float32)
    else:
        assert False, "padding in conv2d_boost wrong"

    input_pointer = fecth_pointer(input_padded)
    output_pointer = fecth_pointer(output_tensor)
    filter_pointer = fecth_pointer(filter)
    assert not boost_conv2d(
        input_pointer,
        output_pointer,
        filter_pointer,
        input_shape[0],  # batch size
        input_shape[1],  # input height
        input_shape[2],  # input width
        input_shape[3],  # input channels
        strides_batch,
        strides_in_height,
        strides_in_width,
        strides_in_channel,
        filter_shape[0],  # filter height
        filter_shape[1],  # filter width
        filter_shape[3]  # output channels
    )
    return output_tensor


def conv2d_gi_boost(sensitivity_map, filter, strides, padding, width_padding_left = 0, width_padding_right = 0,
                    height_padding_up = 0, height_padding_down = 0):
    sensitivity_shape = np.shape(sensitivity_map)  # this_grad
    filter_shape = np.shape(filter)  # filter w
    assert sensitivity_shape[3] == filter_shape[3], "sensitivity_in_channel and filter_out_channel is not equal"
    strides_batch = strides[0]  # strides of batch
    strides_in_height = strides[1]  # strides of height
    strides_in_width = strides[2]  # strides of width
    strides_in_channel = strides[3]  # strides of channel

    if padding == 'SAME':
        sensitivity_padded = np.zeros([sensitivity_shape[0], sensitivity_shape[1] + height_padding_up + height_padding_down,
                                        sensitivity_shape[2] + width_padding_left + width_padding_right, sensitivity_shape[3]], dtype=np.float32)
        sensitivity_padded[:, height_padding_up:sensitivity_shape[1] + height_padding_up,
                            width_padding_left: sensitivity_shape[2] + width_padding_left, :] = sensitivity_map
        result_shape = (sensitivity_shape[0], sensitivity_shape[1], sensitivity_shape[2], filter_shape[2])
        result_tensor = np.ndarray(result_shape, dtype=np.float32)
        sensitivity_shape = np.shape(sensitivity_padded)
    elif padding == 'VALID':
        sensitivity_padded = sensitivity_map
        result_tensor = np.ndarray([sensitivity_shape[0], sensitivity_shape[1] - filter_shape[0] + 1,
                                  sensitivity_shape[2] - filter_shape[1] + 1, filter_shape[3]], dtype=np.float32)
    else:
        assert False, "padding in conv2d_boost_gi wrong"

    sensitivity_pointer = fecth_pointer(sensitivity_padded)
    filter_pointer = fecth_pointer(filter)
    result_pointer = fecth_pointer(result_tensor)
    assert not boost_conv2d_gi(
        sensitivity_pointer,
        filter_pointer,
        result_pointer,
        sensitivity_shape[0],  # batch size
        sensitivity_shape[1],  # sensitivity height
        sensitivity_shape[2],  # sensitivity width
        sensitivity_shape[3],  # sensitivity channels
        strides_batch,
        strides_in_height,
        strides_in_width,
        strides_in_channel,
        filter_shape[0],  # filter height
        filter_shape[1],  # filter width
        filter_shape[2]  # input channels
    )
    return result_tensor

def conv2d_gw_boost(input_tensor, sensitivity_map, strides, padding, width_padding_left = 0, width_padding_right = 0,
                    height_padding_up = 0, height_padding_down = 0):
    input_shape = np.shape(input_tensor)   # input x
    sensitivity_shape = np.shape(sensitivity_map)   # filter w
    strides_batch = strides[0]   # strides of batch
    strides_in_height = strides[1]   # strides of height
    strides_in_width = strides[2]   # strides of width
    strides_in_channel = strides[3]   # strides of channel
    if padding == 'SAME':
        input_padded = np.zeros([input_shape[0], input_shape[1] + height_padding_up + height_padding_down,
                                 input_shape[2] + width_padding_left + width_padding_right, input_shape[3]], dtype=np.float32)
        input_padded[:, height_padding_up:input_shape[1] + height_padding_up,
        width_padding_left: input_shape[2] + width_padding_left, :] = input_tensor
        input_shape = np.shape(input_padded)
        result_shape = (input_shape[1] - sensitivity_shape[1] + 1, input_shape[2] - sensitivity_shape[2] + 1, input_shape[3], sensitivity_shape[3])
        result_tensor = np.ndarray(result_shape, dtype=np.float32)
    elif padding == 'VALID':
        input_padded = input_tensor
        result_shape = (input_shape[1] - sensitivity_shape[1] + 1, input_shape[2] - sensitivity_shape[2] + 1, input_shape[3], sensitivity_shape[3])
        result_tensor = np.ndarray(result_shape, dtype=np.float32)
    else:
        assert False, "padding in conv2d_boost_gw wrong"

    input_pointer = fecth_pointer(input_padded)
    sensitivity_pointer = fecth_pointer(sensitivity_map)
    result_pointer = fecth_pointer(result_tensor)

    assert not boost_conv2d_gw(
        input_pointer,
        sensitivity_pointer,
        result_pointer,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        strides_batch,
        strides_in_height,
        strides_in_width,
        strides_in_channel,
        sensitivity_shape[1],
        sensitivity_shape[2],
        sensitivity_shape[3]
    )
    return result_tensor


def max_pool_boost(input_tensor, k_size, strides, padding):
    input_shape = np.shape(input_tensor)
    strides_batch = strides[0]   # strides of batch
    strides_in_height = strides[1]   # strides of height
    strides_in_width = strides[2]   # strides of width
    strides_in_channel = strides[3]   # strides of channel
    if padding == 'SAME':
        result_shape = (input_shape[0], int(np.ceil(input_shape[1] / strides_in_height)), int(np.ceil(input_shape[2] / strides_in_width)), input_shape[3])
        result_tensor = np.ndarray(result_shape, dtype=np.float32)
    elif padding == 'VALID':
        result_shape = (input_shape[0], np.ceil((input_shape[1] - k_size[1] + 1) / strides_in_height)
                        , np.ceil((input_shape[2] - k_size[2] + 1) / strides_in_width), input_shape[3])
        result_tensor = np.ndarray(result_shape, dtype=np.float32)
    else:
        assert False, "max_pool padding wrong"

    input_pointer = fecth_pointer(input_tensor)
    result_pointer = fecth_pointer(result_tensor)

    assert not boost_max_pool(
        input_pointer,
        result_pointer,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        k_size[1],
        k_size[2],
        result_shape[1],
        result_shape[2],
        strides_batch,
        strides_in_height,
        strides_in_width,
        strides_in_channel,
        )
    return result_tensor


def max_pool_grad_boost(input_tensor, sensitivity_map, k_size, strides):
    input_shape = np.shape(input_tensor)
    strides_batch = strides[0]   # strides of batch
    strides_in_height = strides[1]   # strides of height
    strides_in_width = strides[2]   # strides of width
    strides_in_channel = strides[3]   # strides of channel
    result_tensor = np.zeros(input_shape, dtype=np.float32)
    sensitivity_shape = np.shape(sensitivity_map)

    input_pointer = fecth_pointer(input_tensor)
    sensitivity_pointer = fecth_pointer(sensitivity_map)
    result_pointer = fecth_pointer(result_tensor)

    assert not boost_max_pool_grad(
        input_pointer,
        sensitivity_pointer,
        result_pointer,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        k_size[1],
        k_size[2],
        sensitivity_shape[1],
        sensitivity_shape[2],
        strides_batch,
        strides_in_height,
        strides_in_width,
        strides_in_channel,
    )
    return result_tensor
