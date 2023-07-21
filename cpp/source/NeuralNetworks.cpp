
#include "../headers/NeuralNetworks.h"

float eqFunc(float a)
{
    return a;
}

float unity(float a)
{

    return 1;
}

float uintStepFunc(float a)
{
    if (a > 0)
        return 1;
    return 0;
}

float posLinearFunc(float a)
{
    if (a > 0)
        return a;
    return 0;
}

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float dsigmoid(float y)
{
    return y * (1 - y);
}

Layer::Layer(size_t inputNum, size_t outputNum, float (*func)(float) = eqFunc, float (*dfunc)(float) = unity)
    : weights(outputNum, inputNum), gradiant(outputNum, inputNum), func(func), dfunc(dfunc)
{

#ifdef DEBUG
    printf("Layer created\n");
#endif
}

Layer::~Layer()
{

#ifdef DEBUG
    printf("Layer deleted\n");
#endif
}

void Layer::setWeights(const float *data)
{
    weights.copyData(data);
}

void Layer::generateWeights()
{
    generate(weights);
}

Tensor<float> Layer::calcOutput(Tensor<float> x)
{

    return (weights * x).applyFunction(func);
}

Tensor<float> Layer::calcError(Tensor<float> *x, Tensor<float> *y, size_t num)
{
    Tensor<float> error(y[0].getRowsNum(), y[0].getColumnNum(), 0);

    gradiant.ZERO();

    for (size_t i = 0; i < num; i++)
    {
        Tensor<float> &fz = calcOutput(x[i]);
        Tensor<float> &dfz = fz.applyFunction(dfunc);
        Tensor<float> &e = (y[i] - fz);
        error += e.elementMul(e);

        gradiant += (dfz.elementMul(e)) * x[i].transpose() * -2;
        // printf(gradiant);
    }

    return error;
}

float Layer::learn(Tensor<float> *x, Tensor<float> *y, size_t num, float learningRate = 0.001f, size_t count = 1000u)
{

    float e1 = calcError(x, y, num).norm();
    float e2 = 0;

    for (size_t i = 0; i < count; i++)
    {
        e2 = e1;
        weights += gradiant * (-learningRate);
        e1 = calcError(x, y, num).norm();
        if (e2 - e1 < 0.00001f)
            return e1;
#ifdef DEBUG
        printf("%f \n", e1);
#endif
    }
    return 0;
}

Tensor<float> &Layer::getGradiant()
{
    return gradiant;
}

Tensor<float> &Layer::getWeights()
{
    return weights;
}

void Layer::NumericalCalcGradiant(Tensor<float> *x, Tensor<float> *y, size_t num)
{
    float e, dw = 0.001f;

    for (size_t i = 0; i < weights.getRowsNum(); i++)
    {
        for (size_t j = 0; j < weights.getColumnNum(); j++)
        {
            e = (calcError(x, y, num))[i][j];
            weights[i][j] += dw;

            e -= (calcError(x, y, num))[i][j];
            e /= dw;
            weights[i][j] += dw;
            gradiant[i][j] = e;
        }
    }
}

/*
int main()
{

    Perceptron pr(7);

    float *input[10], y[10], w[8] = {0.1, 0.001, 0.05, -0.03, 0.08, 0.09, 0.0078, 0.025};

    pr.setWeight(w);

    for (short i = 0; i < 10; i++)
    {
        input[i] = new float[7];
        generate(input[i], 7);
        y[i] = float(rand() % 100) / 100;
    }

    for (short i = 0; i < 10; i++)
    {
        float out = pr.calcOutput(input[i]);

        printf("out = %f, y = %f, error = %f\n", out, y[i], abs(out - y[i]));
    }

    pr.learn(y, input, 10);
    printf("e = %f\n", pr.calcErrorAndGradiant(y, input, 10));
    printf(pr.getGrad());
    printf(pr.getWeights());

    for (short i = 0; i < 10; i++)
    {
        float out = pr.calcOutput(input[i]);

        printf("out = %f, y = %f, error = %f\n", out, y[i], abs(out - y[i]));
    }
    printf("e = %f\n", pr.calcErrorAndGradiant(y, input, 10));

    pr.getWeights().insertColumn(0,1);
    pr.getWeights().setColumn(0,w);
    printf(pr.getWeights());

    getchar();
    return 0;
}
*/

int main()
{
    std::vector<Tensor<float>> x, y;
    Tensor<float> a(5, 1);

    generate(a);

    for (size_t i = 0; i < 5; i++)
    {
        x.push_back(Tensor<float>(4, 1));
        generate(x.back());
        y.push_back(a * (-5) + a * (float(rand() % 100) / 1000.0f));
    }

    NeuralNetworks Lr;
    Lr.addLayer(4, 5, sigmoid, dsigmoid);
    Lr.getLayer(0).generateWeights();

    Lr.addLayer(5, 2, sigmoid, dsigmoid);
    Lr.getLayer(1).generateWeights();

    /*
    printf("weights\n");

    printf(Lr.getWeights());

    printf("------------------------------------\n"
            "Numerical gradiant\n");

    Lr.NumericalCalcGradiant(x.data(),y.data(),x.size());
    printf(Lr.getGradiant());

    printf("------------------------------------\n"
            "gradiant\n");

    Lr.learn(x.data(),y.data(),x.size(),0.05,5000);
    printf(Lr.getGradiant());

     printf("------------------------------------\n"
            "weights\n");
    printf(Lr.getWeights());

    */

    printf(Lr.calcOutput(x[0]));
    printf("---------------------------\n");

    getchar();
    return 0;
}
