#include <iostream>

#include "vec.h"


#ifndef API 
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define CAPI extern "C" API
#endif