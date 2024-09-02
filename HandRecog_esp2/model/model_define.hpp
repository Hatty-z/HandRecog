#pragma once
#include <stdint.h>
#include "layer/dl_layer_model.hpp"
#include "layer/dl_layer_base.hpp"
#include "layer/dl_layer_max_pool2d.hpp"
#include "layer/dl_layer_conv2d.hpp"
#include "layer/dl_layer_reshape.hpp"
#include "layer/dl_layer_softmax.hpp"
#include "layer/dl_layer_relu.hpp"
#include "handrecognition_coefficient.hpp"

using namespace dl;
using namespace layer;
using namespace handrecognition_coefficient;

class HANDRECOGNITION : public Model<int16_t> 
{
private:
    Conv2D<int16_t> l1;
    Relu<int16_t> l2;
    MaxPool2D<int16_t> l3;
    Conv2D<int16_t> l4;
    Relu<int16_t> l5; 
    MaxPool2D<int16_t> l6;
    Conv2D<int16_t> l7;
    Relu<int16_t> l8; 
    MaxPool2D<int16_t> l9;
    Reshape<int16_t> l10;
    Conv2D<int16_t> l11;
    Relu<int16_t> l12;
    Conv2D<int16_t> l13;
public:
    Softmax<int16_t> l14; // output layer

    HANDRECOGNITION() : 
    l1(Conv2D<int16_t>(-12, 
                       get_conv_0_filter(), 
                       get_conv_0_bias(), 
                       get_conv_0_activation(), 
                       PADDING_VALID, {}, 1, 1, "l1")),
    l2(Relu<int16_t>("l2")),
    l3(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),                      
    l4(Conv2D<int16_t>(-11, 
                    get_conv_3_filter(), 
                    get_conv_3_bias(), 
                    get_conv_3_activation(), 
                    PADDING_VALID, {}, 1, 1, "l4")),
    l5(Relu<int16_t>("l5")),
    l6(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l6")),
    l7(Conv2D<int16_t>(-9, 
                    get_conv_6_filter(), 
                    get_conv_6_bias(), 
                    get_conv_6_activation(), 
                    PADDING_VALID, {}, 1, 1, "l7")),
    l8(Relu<int16_t>("l8")),
    l9(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l9")),
    l10(Reshape<int16_t>({1, 1, 6400}, "l10")),
    l11(Conv2D<int16_t>(-7, 
                    get_gemm_10_filter(), 
                    get_gemm_10_bias(), 
                    get_gemm_10_activation(), 
                    PADDING_VALID, {}, 1, 1, "l11")),
    l12(Relu<int16_t>("l12")),
    l13(Conv2D<int16_t>(-8, 
                    get_gemm_12_filter(), 
                    get_gemm_12_bias(), 
                    NULL, 
                    PADDING_VALID, {}, 1, 1, "l13")),
    l14(Softmax<int16_t>(-14, "l14")) 
    {}

    void build(Tensor<int16_t> &input)
        {
            this->l1.build(input);
            this->l2.build(this->l1.get_output());
            this->l3.build(this->l2.get_output());
            this->l4.build(this->l3.get_output());
            this->l5.build(this->l4.get_output());
            this->l6.build(this->l5.get_output());
            this->l7.build(this->l6.get_output());
            this->l8.build(this->l7.get_output());
            this->l9.build(this->l8.get_output());
            this->l10.build(this->l9.get_output());
            this->l11.build(this->l10.get_output());
            this->l12.build(this->l11.get_output());
            this->l13.build(this->l12.get_output());
            this->l14.build(this->l13.get_output());       
        }

    void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();

        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();

        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();

        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();

        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();

        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();

        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();

        this->l12.call(this->l11.get_output());
        this->l11.get_output().free_element();

        this->l13.call(this->l12.get_output());
        this->l12.get_output().free_element();

        this->l14.call(this->l13.get_output());
        this->l13.get_output().free_element();
    }
};