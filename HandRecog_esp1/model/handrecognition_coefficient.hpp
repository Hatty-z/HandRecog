#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace handrecognition_coefficient
{
    const dl::Filter<int16_t> *get_conv_0_filter();
    const dl::Bias<int16_t> *get_conv_0_bias();
    const dl::Activation<int16_t> *get_conv_0_activation();
    const dl::Filter<int16_t> *get_conv_3_filter();
    const dl::Bias<int16_t> *get_conv_3_bias();
    const dl::Activation<int16_t> *get_conv_3_activation();
    const dl::Filter<int16_t> *get_conv_6_filter();
    const dl::Bias<int16_t> *get_conv_6_bias();
    const dl::Activation<int16_t> *get_conv_6_activation();
    const dl::Filter<int16_t> *get_gemm_10_filter();
    const dl::Bias<int16_t> *get_gemm_10_bias();
    const dl::Activation<int16_t> *get_gemm_10_activation();
    const dl::Filter<int16_t> *get_gemm_12_filter();
    const dl::Bias<int16_t> *get_gemm_12_bias();
}
