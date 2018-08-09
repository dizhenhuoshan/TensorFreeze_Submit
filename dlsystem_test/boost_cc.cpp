#include <cstdio>
#include <stdlib.h>
#include <cstring>
#include "mkl.h"

inline int find(const int &i, const int &j, const int &k, const int &w, const int &j_size, const int &k_size, const int &w_size)
{
    return ((i * j_size + j) * k_size + k) * w_size + w;
}

inline int find(const int &i, const int &j, const int &k, const int &w, const int &t, const int &j_size, const int &k_size, const int &w_size, const int &t_size)
{
    return (((i * j_size + j) * k_size + k) * w_size + w) * t_size + t;
}

inline int find(const int &i, const int &j, const int &k, const int &w, const int &t, const int &p, const int &j_size, const int &k_size, const int &w_size, const int &t_size, const int &p_size)
{
    return ((((i * j_size + j) * k_size + k) * w_size + w) * t_size + t) * p_size + p;
}

inline float max(const float &a, const float &b)
{
    return a > b ? a : b;
}

inline int mat_find(const int &i, const int &j, const int &j_size)
{
    return i * j_size + j;
}


extern "C"
int matmul(float *mat_left, float *mat_right, float *mat_result, bool trans_left, bool trans_right, int m, int k, int n)
{
    if (trans_left)
    {
        if (trans_right)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, k, 1, mat_left, m, mat_right, k, 0, mat_result, n);
        else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1, mat_left, m, mat_right, n, 0, mat_result, n);
    }
    else
    {
        if (trans_right)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, mat_left, k, mat_right, k, 0, mat_result, n);
        else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, mat_left, k, mat_right, n, 0, mat_result, n);
    }
    return 0;
}

extern "C"
int relu(float* input_tensor, float *result, int size)
{
    register int i = 0;
    for (i = 0; i < size; i++)
        *(result + i) = max(*(input_tensor + i), 0);
    return 0;
}

extern "C"
int relu_grad(float *relu_output, float *result, int size)
{
    register int i = 0;
    for (i = 0; i < size; i++)
        *(result + i) = *(relu_output + i) > 0;
    return 0;
}

extern "C"
int conv2d(float* input_tensor, float *output_tensor, float* filter,
           int batch_size, int input_height, int input_width, int input_channels,
           int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
           int filter_height, int filter_width, int output_channels)
{
    
    register int b = 0; // batch number
    int f = 0; // filter number/output_channel number
    int c = 0; // input_channel number
    int si_max = input_height - filter_height + 1;
    int sj_max = input_width - filter_width + 1;
    int si = 0; // the i_pos of the left_up block of filter in the input
    int sj = 0; // the j_pos of the left_up block of filter in the input
    int ci = 0; // the current calculate block i_pos in the filter
    int cj = 0; // the current calculate block j_pos in the filter
    float *input_flatten = (float *)malloc(batch_size * filter_height * filter_width * input_channels * si_max * sj_max * sizeof(float));
    for (b = 0; b < batch_size; b += stride_batch)
    {
        for (si = 0; si < si_max; si += stride_in_height)
        {
            for (sj = 0; sj < sj_max; sj += stride_in_width)
            {
                for (ci = 0; ci < filter_height; ci++)
                {
                    memcpy(input_flatten + find(b, si, sj, ci, 0, si_max, sj_max, filter_height, filter_width * input_channels),
                           input_tensor + find(b, si + ci, sj, 0, input_height, input_width, input_channels),
                            filter_width * input_channels * sizeof(float));
                }
            }
        }
    }
    // puts("enter matmul in conv2d");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size * si_max * sj_max, output_channels,
            filter_height * filter_width * input_channels, 1, input_flatten,
            filter_height * filter_width * input_channels, filter, output_channels, 0, output_tensor, output_channels);
    // puts("exit matmul in conv2d\n");
    free(input_flatten);
    return 0;
}

extern "C"
int conv2d_gi(float* sensitivity_map, float* filter, float* result,
                    int batch_size, int sensitivity_height, int sensitivity_width, int sensitivity_channels,
                    int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
                    int filter_height, int filter_width, int input_channels)
{
    
    int b = 0; // batch number
    int f = 0; // filter number/sensitivity_channel number
    int c = 0; // filter_input_channel/result_channel number
    int si_max = sensitivity_height - filter_height + 1;
    int sj_max = sensitivity_width - filter_width + 1;
    int si = 0; // the i_pos of the left_up block of filter in the sensitivity_map
    int sj = 0; // the j_pos of the left_up block of filter in the sensitivity_map
    int ci = 0; // the current calculate block i_pos in the filter, need to reverse
    int cj = 0; // the current calculate block j_pos in the filter, need to reverse
    float *sensitivity_flatten = (float*)malloc(batch_size * si_max * sj_max * filter_height * filter_width * sensitivity_channels * sizeof(float));
    float *filter_flatten = (float*)malloc(filter_height * filter_width * input_channels * sensitivity_channels * sizeof(float));

    for (ci = 0; ci < filter_height; ci++)
        for (cj = 0; cj < filter_width; cj++)
            for (c = 0; c < input_channels; c++)
                for (f = 0; f < sensitivity_channels; f++)
                    *(filter_flatten + find(ci, cj, f, c, filter_width, sensitivity_channels, input_channels))
                    = *(filter + find(filter_height - ci - 1, filter_width - cj - 1, c, f, filter_width, input_channels, sensitivity_channels));

    for (b = 0; b < batch_size; b += stride_batch)
    {
        for (si = 0; si < si_max; si += stride_in_height)
        {
            for (sj = 0; sj < sj_max; sj += stride_in_width)
            {
                for (ci = 0; ci < filter_height; ci++)
                {
                    memcpy(sensitivity_flatten + find(b, si, sj, ci, 0, si_max, sj_max, filter_height, filter_width * sensitivity_channels),
                           sensitivity_map + find(b, si + ci, sj, 0, sensitivity_height, sensitivity_width, sensitivity_channels),
                           filter_width * sensitivity_channels * sizeof(float));
                }
            }
        }
    }
    // puts("enter matmul in conv2d_gi");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            batch_size * si_max * sj_max, input_channels, filter_height * filter_width * sensitivity_channels,
            1, sensitivity_flatten, filter_height * filter_width * sensitivity_channels,
           filter_flatten, input_channels, 0, result, input_channels);
    // puts("exit matmul in conv2d_gi\n");
    free(sensitivity_flatten);
    free(filter_flatten);
    return 0;
}

extern "C"
int conv2d_gw(float *input_tensor, float *sensitivity_map, float* result,
            int batch_size, int input_height, int input_width, int input_channels,
            int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
            int sensitivity_height, int sensitivity_width, int sensitivity_channels)
{

    int b = 0; // batch number
    int f = 0; // sensitivity_channel number
    int c = 0; // input_channel
    int ci_max = input_height - sensitivity_height + 1; // filter height
    int cj_max = input_width - sensitivity_width + 1; // filter width
    int ci = 0; // the i_pos of the left_up block of sensitivity_map in the input_padding
    int cj = 0; // the j_pos of the left_up block of sensitivity_map in the input_padding
    int si = 0; // the current calculate block i_pos in the sensitivity_map
    int sj = 0; // the current calculate block j_pos in the sensitivity_map
    float *input_flatten = (float*)malloc(ci_max * cj_max * input_channels * batch_size * sensitivity_height * sensitivity_width * sizeof(float));
    for (b = 0; b < batch_size; b += stride_batch)
    {
        for (ci = 0; ci < ci_max; ci++)
        {
            for (cj = 0; cj < cj_max; cj++)
            {
                for (si = 0; si < sensitivity_height; si += stride_in_height)
                {
                    for (sj = 0; sj < sensitivity_width; sj += stride_in_width)
                    {
                        for (c = 0; c < input_channels; c++)
                        {
                            *(input_flatten +
                               find(ci, cj, c, b, si, sj, cj_max, input_channels, batch_size, sensitivity_height, sensitivity_width))
                               = *(input_tensor + find(b, si + ci, sj + cj, c, input_height, input_width, input_channels));
                        }
                    }
                }
            }
        }
    }
//    puts("enter conv2d_gw");
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            ci_max * cj_max * input_channels, sensitivity_channels, batch_size * sensitivity_height * sensitivity_width
            , 1, input_flatten, batch_size * sensitivity_height * sensitivity_width, sensitivity_map,
            sensitivity_channels, 0, result, sensitivity_channels);
//    puts("exit conv2d_gw");
    free(input_flatten);
    return 0;
}

extern "C"
int conv2d_gw_right(float *input_tensor, float *sensitivity_map, float* result,
              int batch_size, int input_height, int input_width, int input_channels,
              int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
              int sensitivity_height, int sensitivity_width, int sensitivity_channels)
{
    int b = 0; // batch number
    int f = 0; // sensitivity_channel number
    int c = 0; // input_channel
    int ci_max = input_height - sensitivity_height + 1;
    int cj_max = input_width - sensitivity_width + 1;
    int ci = 0; // the i_pos of the left_up block of sensitivity_map in the input_padding
    int cj = 0; // the j_pos of the left_up block of sensitivity_map in the input_padding
    int si = 0; // the current calculate block i_pos in the sensitivity_map
    int sj = 0; // the current calculate block j_pos in the sensitivity_map
    for (b = 0; b < batch_size; b += stride_batch)
        for (ci = 0; ci < ci_max; ci++)
            for (cj = 0; cj < cj_max; cj++)
                for (si = 0; si < sensitivity_height; si += stride_in_height)
                    for (sj = 0; sj < sensitivity_width; sj += stride_in_width)
                        for (c = 0; c < input_channels; c += stride_in_channel)
                            for (f = 0; f < sensitivity_channels; f++)
                                *(result + find(ci, cj, c, f, cj_max, input_channels, sensitivity_channels)) +=
                                        *(input_tensor + find(b, ci + si, cj + sj, c, input_height, input_width, input_channels))
                                        * (*(sensitivity_map + find(b, si, sj, f, sensitivity_height, sensitivity_width, sensitivity_channels)));

    return 0;
}

extern "C"
int max_pool(float* input_tensor, float *result,
             int batch_size, int input_height, int input_width, int input_channels,
             int windows_height, int windows_width, int result_height, int result_width,
             int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel)
{
    int b = 0; //batch number
    int c = 0; //channel number
    int si = 0; //si is the i_pos of the left_up block of window in the input
    int sj = 0; //si is the j_pos of the left_up block of window in the input
    int si_max = input_height - windows_height + 1;
    int sj_max = input_width - windows_width + 1;
    int ci = 0; //ci is the i_pos of the current block in the window
    int cj = 0; //cj is the j_pos of the current block in the window
    float tmp_max = 0; // tmp_max is to find the max number
    for (b = 0; b < batch_size; b += stride_batch)
    {
        for (c = 0; c < input_channels; c += stride_in_channel)
        {
            for (si = 0; si < si_max; si += stride_in_height)
            {
                for (sj = 0; sj < sj_max; sj += stride_in_width)
                {
                    tmp_max = *(input_tensor + find(b, si, sj, c, input_height, input_width, input_channels));
                    for (ci = 0; ci < windows_height; ci++)
                    {
                        for (cj = 0; cj < windows_width; cj++)
                        {
                            tmp_max = max(tmp_max, *(input_tensor + find(b, si + ci, sj + cj, c, input_height, input_width, input_channels)));
                        }
                    }
                    *(result + find(b, si / stride_in_height, sj / stride_in_width, c, result_height, result_width, input_channels))
                    = tmp_max;
                }
            }
        }
    }
    return 0;
}

extern "C"
int max_pool_grad(float *input_tensor, float *sensitivity_map, float *result,
                  int batch_size, int input_height, int input_width, int input_channels,
                  int windows_height, int windows_width, int sensitivity_height, int sensitivity_width,
                  int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel)
{
    int b = 0; //batch number
    int c = 0; //channel number
    int si = 0; //si is the i_pos of the left_up block of window in the input
    int sj = 0; //si is the j_pos of the left_up block of window in the input
    int si_max = input_height - windows_height + 1;
    int sj_max = input_width - windows_width + 1;
    int ci = 0; //ci is the i_pos of the current block in the window
    int cj = 0; //cj is the j_pos of the current block in the window
    int max_ci = 0; //max_ci is the pos_i of max element in the current window
    int max_cj = 0; //max_ci is the pos_j of max element in the current window
    for (b = 0; b < batch_size; b += stride_batch)
    {
        for (c = 0; c < input_channels; c += stride_in_channel)
        {
            for (si = 0; si < si_max; si += stride_in_height)
            {
                for (sj = 0; sj < sj_max; sj += stride_in_width)
                {
                    max_ci = 0; max_cj = 0;
                    for (ci = 0; ci < windows_height; ci++)
                    {
                        for (cj = 0; cj < windows_width; cj++)
                        {
                            if (* (input_tensor + find(b, si + max_ci, sj + max_cj, c, input_height, input_width, input_channels))
                                < *(input_tensor + find(b, si + ci, sj + cj, c, input_height, input_width, input_channels)))
                            {
                                max_ci = ci; max_cj = cj;
                            }
                        }
                    }
                    *(result + find(b, si + max_ci, sj + max_cj, c, input_height, input_width, input_channels))
                            = *(sensitivity_map + find(b, si / stride_in_height, sj / stride_in_width, c, sensitivity_height, sensitivity_width, input_channels));
                }
            }
        }
    }
    return 0;
}

extern "C"
int conv2d_right(float* input_tensor, float *output_tensor, float* filter,
                 int batch_size, int input_height, int input_width, int input_channels,
                 int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
                 int filter_height, int filter_width, int output_channels)
{
    int b = 0; // batch number
    int f = 0; // filter number/output_channel number
    int c = 0; // input_channel number
    int si_max = input_height - filter_height + 1;
    int sj_max = input_width - filter_width + 1;
    int si = 0; // the i_pos of the left_up block of filter in the input
    int sj = 0; // the j_pos of the left_up block of filter in the input
    int ci = 0; // the current calculate block i_pos in the filter
    int cj = 0; // the current calculate block j_pos in the filter
    for (b = 0; b < batch_size; b += stride_batch)
        for (si = 0; si < si_max; si += stride_in_height)
            for (sj = 0; sj < sj_max; sj += stride_in_width)
                for (ci = 0; ci < filter_height; ci++)
                    for (cj = 0; cj < filter_width; cj++)
                        for (c = 0; c < input_channels; c += stride_in_channel)
                            for (f = 0; f < output_channels; f++)
                                *(output_tensor + find(b, si, sj, f, si_max, sj_max, output_channels)) +=
                                        *(input_tensor + find(b, si + ci, sj + cj, c, input_height, input_width, input_channels))
                                        * (*(filter + find(ci, cj, c, f, filter_width, input_channels, output_channels)));
    return 0;
}

extern "C"
int conv2d_gi_right(float* sensitivity_map, float* filter, float* result,
                    int batch_size, int sensitivity_height, int sensitivity_width, int sensitivity_channels,
                    int stride_batch, int stride_in_height, int stride_in_width, int stride_in_channel,
                    int filter_height, int filter_width, int input_channels)
{
    int b = 0; // batch number
    int f = 0; // filter number/sensitivity_channel number
    int c = 0; // filter_input_channel/result_channel number
    int si_max = sensitivity_height - filter_height + 1;
    int sj_max = sensitivity_width - filter_width + 1;
    int si = 0; // the i_pos of the left_up block of filter in the sensitivity_map
    int sj = 0; // the j_pos of the left_up block of filter in the sensitivity_map
    int ci = 0; // the current calculate block i_pos in the filter, need to reverse
    int cj = 0; // the current calculate block j_pos in the filter, need to reverse
    for (b = 0; b < batch_size; b += stride_batch)
        for (si = 0; si < si_max; si += stride_in_height)
            for (sj = 0; sj < sj_max; sj += stride_in_width)
                for (ci = 0; ci < filter_height; ci++) //reverse
                    for (cj = 0; cj < filter_width; cj++) //reverse
                        for (c = 0; c < input_channels; c += stride_in_channel)
                            for (f = 0; f < sensitivity_channels; f++)
                                *(result + find(b, si, sj, c, si_max, sj_max, input_channels)) +=
                                        *(sensitivity_map + find(b, si + ci, sj + cj, f, sensitivity_height, sensitivity_width, sensitivity_channels))
                                        * ( *(filter + find(ci, cj, c, f, filter_width, input_channels, sensitivity_channels)));

    return 0;
}
