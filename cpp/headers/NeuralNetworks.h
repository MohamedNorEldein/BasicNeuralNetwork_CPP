#pragma once
#include <vector>

#define DEBUG
#include "vec.h"



float eqFunc(float);
float uintStepFunc(float);
float posLinearFunc(float);
float sigmoid(float);
float dsigmoid(float);
float unity(float a);

class Perceptron
{
private:
    Tensor<float> weights;
    float (*func)(float);
    float (*dfunc)(float);

private:
    float error = 0;
    Tensor<float> grad;

public:
    Perceptron(size_t inputNum) : weights(inputNum + 1, 1), func(sigmoid), dfunc(dsigmoid), grad(inputNum + 1, 1)
    {
    }

    ~Perceptron()
    {
    }

public:
    float calcOutput(const float *inputData)
    {
        Tensor<float> input(1, weights.getRowsNum() - 1, inputData);
        input.insertColumn(0, 1);

        return func((input * weights).getElement(0, 0));
    }

    void setWeight(const float *data)
    {
        weights.copyData(data);
    }

    void generateWeights()
    {
        generate(weights);
        printf(weights);
    }

public:
    float calcErrorAndGradiant(float *y, float **x, size_t num)
    {
        float fz = 0, dfz = 0, e = 0;
        Tensor<float> Tx(grad.getRowsNum(), grad.getColumnNum(), 0);
        error = 0;
        grad.ZERO();

        for (size_t i = 0; i < num; i++)
        {
            fz = calcOutput(x[i]);
            dfz = dfunc(fz);
            e = y[i] - fz;
            error += e * e;
            Tx.copyData(x[i]);
            grad += Tx * 2 * dfz * e;
        }

        return error;
    }

    float learn(float *y, float **x, size_t num)
    {

        float e1 = calcErrorAndGradiant(y, x, num);
        float e2 = 0;
        float s = 0.01;

        for (size_t i = 0; i < 1000; i++)
        {
            weights += grad * (-s);
            e2 = calcErrorAndGradiant(y, x, num);
            // printf(weights);
            if (abs(e2 - e1) < 0.000001)
                return abs(e2 - e1);
            e1 = e2;
        }

        return abs(e2 - e1);
    }

    Tensor<float> &getGrad()
    {
        return grad;
    }
    Tensor<float> &getWeights()
    {
        return weights;
    }
};


class Layer
{
private:
    Tensor<float> weights, gradiant;
    float (*func)(float);
    float (*dfunc)(float);

public:
    Layer(size_t inputNum, size_t outputNum,float (*func)(float),float (*dfunc)(float) );
    ~Layer();

private:
public:
    void setWeights(const float *data);
    Tensor<float>& getWeights();
    Tensor<float>& getGradiant();


    void generateWeights();

    Tensor<float> calcOutput(Tensor<float> x);
    
    Tensor<float> calcError(Tensor<float> *x, Tensor<float> *y, size_t num);
    void NumericalCalcGradiant(Tensor<float> *x, Tensor<float> *y, size_t num);
    
    float learn(Tensor<float> *x, Tensor<float> *y, size_t num, float learningRate,size_t count );
};


class NeuralNetworks
{
private:
    std::vector<Layer> layers;


public:
    NeuralNetworks():
        layers()
    {

    }
    ~NeuralNetworks(){

    }

public:
    void addLayer(size_t inputNum, size_t outputNum,float (*func)(float),float (*dfunc)(float) )
    {
        layers.push_back(Layer(inputNum, outputNum,func,dfunc));
    }


    Layer& getLayer(size_t i){
        return layers[i];
    }

    Tensor<float> calcOutput(Tensor<float> x){
        Tensor<float> y=x;
        printf("layers.size() = %u\n", layers.size());
        for (size_t i = 0; i < layers.size(); i++)
        {
            printf(y);
            y.set( layers[i].calcOutput(y));
        }
        
        return y;
    }


};
