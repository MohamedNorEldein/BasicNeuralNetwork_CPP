#pragma once
#include <vector>

#include "vec.h"
#include "CAPI.h"

#define TensorFloat  Tensor<float>

float eqFunc(float);
float uintStepFunc(float);
float posLinearFunc(float);
float sigmoid(float);
float dsigmoid(float);
float unity(float a);

class NeuralNetworks
{
public:
    std::vector<TensorFloat> weights, weights_gradints,err, outputs, bias, bias_gradints;
    size_t inputNum, outNum;
    std::vector<float (*)(float)> funcs;
    std::vector<float (*)(float)> dfuncs;

public:
    NeuralNetworks(size_t inputNum);
    ~NeuralNetworks();

public:
    void addLayer(size_t outputNum, float (*func)(float), float (*dfunc)(float),float*a =0);

    void print(std::vector<TensorFloat> &ws);
public: 
    TensorFloat& getWeight(size_t index){
        return weights[index];
    }

    std::vector<TensorFloat>& getWeights(){
        return weights;
    }
    
public:
    TensorFloat calcOutput(const TensorFloat &x);

    TensorFloat calcError(const TensorFloat &x, const TensorFloat &y);

private:
    void backwordProbagration(const TensorFloat &x, const TensorFloat &y);

    void forwardBass(const TensorFloat &x, const TensorFloat &y);
    void update(float);
    void ZERO();


public:
    float Probagration(float **_x, float **_y, size_t n, float learningRate);

    float Probagration(float *_x, float *_y, size_t n, float learningRate);

    float test(float **test_x, float **test_y, size_t test_n);

    float test(float *test_x, float *test_y, size_t test_n);

public:
    float train(float **learn_x, float **learn_y, size_t n, size_t count, float learningRate);
    float train(float *learn_x, float *learn_y, size_t n, size_t count, float learningRate);

    float learn(float **train_x, float **train_y, size_t train_n, float **test_x, float **test_y, size_t test_n, size_t count, float learningRate);
    float learn(float *train_x, float *train_y, size_t train_n, float *test_x, float *test_y, size_t test_n, size_t count, float learningRate);

};
