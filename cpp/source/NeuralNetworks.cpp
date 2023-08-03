
#include "../headers/NeuralNetworks.h"

float eqFunc(float a)
{
    return a;
}

float unity(float a)
{

    return 1;
}

float dRelu(float a)
{
    if (a > 0)
        return 1;
    return 0.00001;
}

float relu(float a)
{
    if (a > 0)
        return a;
    return 0.00001 * a;
}

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float dsigmoid(float y)
{
    return y * (1 - y);
}

float ftanh(float x)
{
    return tanh(x);
}

float dtanh(float x)
{
    return 1 - x * x;
}


float func1(float y, float fz)
{
    return (y - fz);
}

float func2(float y, float fz)
{
    float r = abs(y - fz);
   return (1 / (1.00001f - r)) * (y - fz);
#ifdef DEBUG_NEURAL_NETWORK
    // printf("\tr = %f, er = %f\n", r, result[i][0]);
#endif
}

NeuralNetworks::NeuralNetworks(size_t inputNum) : weights(), weights_gradints(), err(), outputs(), bias(), bias_gradints(),
                                                  inputNum(inputNum), outNum(inputNum),errorFunction(func1)
{
}
NeuralNetworks::~NeuralNetworks()
{
}

void NeuralNetworks::addLayer(size_t outputNum, float (*func)(float), float (*dfunc)(float), float *data)
{
    weights.push_back(TensorFloat(outputNum, outNum));
    weights_gradints.push_back(TensorFloat(outputNum, outNum));

    err.push_back(TensorFloat(outputNum, 1, 0));
    outputs.push_back(TensorFloat(outputNum, 1, 0));

    bias.push_back(TensorFloat(outputNum, 1, 0));
    bias_gradints.push_back(TensorFloat(outputNum, 1, 0));

    funcs.push_back(func);
    dfuncs.push_back(dfunc);

    outNum = outputNum;

    bias.back().ZERO();

    if (!data)
        weights.back().ZERO();
    else
        weights.back().copyData(data);
}

void NeuralNetworks::print(std::vector<TensorFloat> &ws)
{
    for (auto a : ws)
        printf(a);
}

TensorFloat NeuralNetworks::calcOutput(const TensorFloat &x)
{
    outputs[0] = (weights[0] * x + bias[0]).applyFunction(funcs[0]);

    for (size_t i = 1; i < outputs.size(); i++)
    {
        outputs[i] = (weights[i] * outputs[i - 1] + bias[i]).applyFunction(funcs[i]);
    }

    return outputs.back();
}


TensorFloat ErrorFunction(const TensorFloat &y, const TensorFloat &fz, float (*func)(float, float))
{
    size_t n = y.getRowsNum();
    TensorFloat result(n, 1);
    float r;

    for (size_t i = 0; i < n; i++)
    {
        result[i][0] = func(y[i][0], fz[i][0]);
    }

    return result;
}

TensorFloat NeuralNetworks::calcError(const TensorFloat &x, const TensorFloat &y)
{
    return ErrorFunction(y, calcOutput(x),errorFunction);
}

void NeuralNetworks::backwordProbagration(const TensorFloat &x, const TensorFloat &y)
{
    TensorFloat error(calcError(x, y));

    err[outputs.size() - 1] = error.elementMul(outputs.back().applyFunction(dfuncs.back()));

    for (size_t i = outputs.size() - 1; i > 0; --i)
    {
        err[i - 1] = ((weights[i].transpose() * err[i]).elementMul(outputs[i - 1].applyFunction(dfuncs[i - 1])));
    }
}

void NeuralNetworks::forwardBass(const TensorFloat &x, const TensorFloat &y)
{
    weights_gradints[0] += err[0] * x.transpose();
    bias_gradints[0] += err[0];

    for (size_t i = 1; i < weights.size(); i++)
    {
        weights_gradints[i] += err[i] * outputs[i - 1].transpose();
        bias_gradints[i] += err[i];
    }
}

void NeuralNetworks::update(float learning)
{

    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] += weights_gradints[i] * (learning);
        bias[i] += bias_gradints[i] * (learning);
    }
}

void NeuralNetworks::ZERO()
{

    for (size_t i = 0; i < weights.size(); i++)
    {
        weights_gradints[i].ZERO();
        bias_gradints[i].ZERO();
    }
}

float NeuralNetworks::Probagration(float **_x, float **_y, size_t n, float learningRate)
{
    TensorFloat x, y;
    float er = 0;

    ZERO();
    for (size_t i = 0; i < n; i++)
    {
        x.setData(_x[i], inputNum, 1);
        y.setData(_y[i], outNum, 1);

        backwordProbagration(x, y);
        forwardBass(x, y);

        er += (y - outputs.back()).norm();

        x.setData(0, 0, 0);
        y.setData(0, 0, 0);
    }
    update(learningRate / n);
    return er;
}

float NeuralNetworks::Probagration(float *_x, float *_y, size_t n, float learningRate)
{
    TensorFloat x, y;
    float er = 0;
    ZERO();
    for (size_t i = 0; i < n; i++)
    {
        x.setData(_x, inputNum, 1);
        y.setData(_y, outNum, 1);

        backwordProbagration(x, y);
        forwardBass(x, y);

        er += (y - outputs.back()).norm();

        x.setData(0, 0, 0);
        y.setData(0, 0, 0);
        _x += inputNum;
        _y += outNum;
    }
    update(learningRate / n);

    return er;
}

float NeuralNetworks::train(float **learn_x, float **learn_y, size_t n, size_t count, float learningRate)
{
    float er, e2 = 0;
    size_t c = size_t(sqrt(float(count)));

    for (size_t i = 0; i < c; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            er = Probagration(learn_x, learn_y, n, learningRate);
        }
        printf("%f\n", er);
        // if(abs(er-e2)<er/100)
        //    return er;
        e2 = er;
    }
    return er;
}

float NeuralNetworks::train(float *learn_x, float *learn_y, size_t n, size_t count, float learningRate)
{
    float er, e2 = 0;
    size_t c = size_t(sqrt(float(count)));

    for (size_t i = 0; i < c; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            er = Probagration(learn_x, learn_y, n, learningRate);
        }
#ifdef DEBUG_NEURAL_NETWORK
        printf("%f, %f \n", er, er / n);
#endif
        //  if(abs(er-e2)<er/100)
        //     return er;
        e2 = er;
    }
    return er;
}

float NeuralNetworks::test(float **test_x, float **test_y, size_t test_n)
{
    TensorFloat x, y;
    float error = 0;
    for (size_t i = 0; i < test_n; i++)
    {
        x.setData(test_x[i], inputNum, 1);
        y.setData(test_y[i], outNum, 1);

        error += calcError(x, y).norm();

        x.setData(0, 0, 0);
        y.setData(0, 0, 0);
    }
    return error;
}

float NeuralNetworks::test(float *test_x, float *test_y, size_t test_n)
{
    TensorFloat x, y;
    float error = 0;
    for (size_t i = 0; i < test_n; i++)
    {
        x.setData(test_x, inputNum, 1);
        y.setData(test_y, outNum, 1);

        error += calcError(x, y).norm();

        x.setData(0, 0, 0);
        y.setData(0, 0, 0);

        test_x += inputNum;
        test_y += outNum;
    }
    return error;
}

float NeuralNetworks::learn(float **train_x, float **train_y, size_t train_n, float **test_x, float **test_y, size_t test_n, size_t count, float learningRate)
{
    float er, e2 = 0, e1 = 0;
    size_t c = size_t(sqrt(float(count)));

    for (size_t i = 0; i < c; i++)
    {

        for (size_t j = 0; j < c; j++)
        {
            er = Probagration(train_x, train_y, train_n, learningRate);
        }
        e1 = test(test_x, test_y, test_n);
        printf("%f\t%f\n", er, e1);
    }
    return er;
}

float NeuralNetworks::learn(float *train_x, float *train_y, size_t train_n, float *test_x, float *test_y, size_t test_n, size_t count, float learningRate)
{
    float er, e2 = 0, e1 = 0;
    size_t c = size_t(sqrt(float(count)));

    for (size_t i = 0; i < c; i++)
    {

        for (size_t j = 0; j < c; j++)
        {
            er = Probagration(train_x, train_y, train_n, learningRate);
        }
        e1 = test(test_x, test_y, test_n);
        printf("%f\t%f\n", er, e1);
    }
    return er;
}

    void NeuralNetworks::setErrorFunction(float (*func_new)(float, float)){
        this->errorFunction = func_new;
    }



int main()
{
    const size_t count = 50, a = 4, b = 2;
    float *x[count], *y[count];

    for (size_t i = 0; i < count; i++)
    {
        x[i] = new float[a];
        generate(x[i], a);

        y[i] = new float[b];
        y[i][0] = sigmoid((x[i][0] - x[i][1]));
        y[i][1] = sigmoid((x[i][2] - x[i][3]));
    }

    float w1[20] = {
        0.0505121, 0.0499083, 0.108015, 0.0369231,
        0.0618518, 0.079544, 0.0213861, 0.0827119,
        0.0623513, -0.00775296, 0.0132197, 0.0216701,
        0.072, 0.0428035, 0.0362337, 0.0380796,
        0.0684908, 0.0546522, 0.0988076, 0.0150585};
    float w2[10] = {
        0.119051, 0.102308, 0.101671, 0.0565317, 0.125,
        0.124907, 0.0644523, 0.0820344, 0.0625001, 0.123858};
    float w3[4] = {
        -0.465759, -0.450063,
        -0.260323, -0.271469};

    NeuralNetworks Lr(a);
    Lr.addLayer(5, eqFunc, unity);
    Lr.addLayer(2, sigmoid, dsigmoid);

    Lr.addLayer(b, sigmoid, dsigmoid);

    generate(Lr.getWeight(0));
    generate(Lr.weights[1]);
    generate(Lr.weights[2]);

    printf(Lr.bias[0]);
    printf(Lr.bias[1]);
    printf(Lr.bias[2]);

    printf("%f\n", Lr.learn(x, y, count * 4 / 5, x + count * 4 / 5, y + count * 4 / 5, count * 1 / 5, 1000000, 0.1));

    printf("*********************************************\n");

    for (size_t i = 0; i < count; i++)
    {

        printf("fz[%u]\t:", i);
        print(Lr.calcError(TensorFloat(a, 1, x[i]), TensorFloat(b, 1, y[i])).getData(), b);
        printf("------------------------------------\n");
    }
    printf(Lr.bias[0]);
    printf(Lr.bias[1]);
    printf(Lr.bias[2]);

    getchar();
    return 0;
}
