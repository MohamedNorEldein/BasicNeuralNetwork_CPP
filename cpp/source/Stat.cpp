
#include "stats.h"

/*

int main()
{

    Tensor<float> a(4,4);
    generate(a);
    Tensor<float>& b =a.inverse();

    printf(a);
    printf(b*a);
    printf(a+b*a*(a*2-b));

    a.insertColumn(0,1);
    printf(a);

    a.insertRow(5,1);
    printf(a);

    getchar();
    return 0;
}
*/
#define pTensor Tensor<float> *

CAPI void *createTensorFloat(size_t rowNum, size_t columnNum)
{
    pTensor tf = new Tensor<float>(rowNum, columnNum);
    generate(*tf);
    return tf;
}

CAPI void *initTensorFloat(size_t rowNum, size_t columnNum, float *data)
{
    pTensor tf = new Tensor<float>(rowNum, columnNum, data);

    return tf;
}

CAPI void deleteTensorFloat(void *tf)
{
    delete ((pTensor)tf);
}

CAPI void *addTensorFloat(void *tf1, void *tf2)
{
    pTensor result = new Tensor<float>(((pTensor)tf1)->getRowsNum(), ((pTensor)tf1)->getColumnNum());
    Tensor<float>::add((pTensor)tf1, (pTensor)tf2, result);
#ifdef DEBUG
    printf("%x, %x, %x\n", tf1, tf2, result);
#endif

    return result;
}

CAPI void *subTensorFloat(void *tf1, void *tf2)
{
    pTensor result = new Tensor<float>(((pTensor)tf1)->getRowsNum(), ((pTensor)tf1)->getColumnNum());
    Tensor<float>::sub((pTensor)tf1, (pTensor)tf2, result);
    return result;
}

CAPI void *mullTensorFloat(void *tf1, void *tf2)
{
    pTensor result = new Tensor<float>(((pTensor)tf1)->getRowsNum(), ((pTensor)tf2)->getColumnNum());
    Tensor<float>::matmull((pTensor)tf1, (pTensor)tf2, result);
    return result;
}


CAPI void *transposeTensorFloat(void *tf1)
{
    pTensor result = new Tensor<float>(((pTensor)tf1)->getRowsNum(), ((pTensor)tf1)->getColumnNum());
    Tensor<float>::transpose((pTensor)tf1, result);
    return result;
}


CAPI void *inverseTensorFloat(void *tf1)
{
    pTensor result = new Tensor<float>(((pTensor)tf1)->inverse());
    
    return result;
}

CAPI void print(void *tf)
{
    printf(*(pTensor)tf);
}


CAPI void setElementTensorFloat(void *tf, size_t i, size_t j, float value)
{
    ((pTensor)tf)->setElement(i,j,value);
}


CAPI float getElementTensorFloat(void *tf, size_t i, size_t j)
{   
#ifdef DEBUG
    printf("%u, %u\n", i,j);
#endif
    return ((pTensor)tf)->getElement(i,j);
}
