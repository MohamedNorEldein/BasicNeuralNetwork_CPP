
#pragma once
#include "vec.h"
#include "stats.h"


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


#include "../headers/stats.h"

#define pTensor Tensor<float> *

CAPI void *createTensorFloat(size_t rowNum, size_t columnNum);

CAPI void *initTensorFloat(size_t rowNum, size_t columnNum, float *data);

CAPI void deleteTensorFloat(void *tf);

CAPI void *addTensorFloat(void *tf1, void *tf2);

CAPI void *subTensorFloat(void *tf1, void *tf2);

CAPI void *mullTensorFloat(void *tf1, void *tf2);

CAPI void *transposeTensorFloat(void *tf1);

CAPI void *inverseTensorFloat(void *tf1);

CAPI void print(void *tf);

CAPI void setElementTensorFloat(void *tf, size_t i, size_t j, float value);

CAPI float getElementTensorFloat(void *tf, size_t i, size_t j);

CAPI void *getCorelationMatrix(void *a);