
#pragma once
#include "vec.h"
#include "stats.h"
#include "NeuralNetworks.h"

#ifndef API
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define CAPI extern "C" API
#else
#define CAPI API
#endif


#define pTensor Tensor<float> *

CAPI void *createTensorFloat(size_t rowNum, size_t columnNum);

CAPI void *initTensorFloat(size_t rowNum, size_t columnNum, float *data);

CAPI void deleteTensorFloat(void *tf);

CAPI void *addTensorFloat(void *tf1, void *tf2);

CAPI void *subTensorFloat(void *tf1, void *tf2);

CAPI void *mullTensorFloat(void *tf1, void *tf2);

CAPI void *transposeTensorFloat(void *tf1);

CAPI void *inverseTensorFloat(void *tf1);

CAPI void printTensorFloat(void *tf);

CAPI void setElementTensorFloat(void *tf, size_t i, size_t j, float value);

CAPI float getElementTensorFloat(void *tf, size_t i, size_t j);

CAPI void *getCorelationMatrix(void *a);


CAPI void *CreateNeuralNetwork(size_t inputNum);
CAPI void DeleteNeuralNetwork(void *);
CAPI void addLayerNeuralNetwork(void *, size_t,void*);
CAPI void addLogisticLayerNeuralNetwork(void *pNeuralNetwork, size_t outputNum, void *data);
CAPI void* calcOutputNeuralNetwork(void *, void*);
CAPI void* calcErrorNeuralNetwork(void *,  void*,  void *);

CAPI float trainNeuralNetwork(void *, void *, void *, size_t, size_t, float);
CAPI float learnNeuralNetwork(void *, void *, void *, size_t, void *, void *, size_t, size_t, float);
