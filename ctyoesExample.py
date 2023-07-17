"""
    this script is an application of ctypes liberary in python where the linear algebra class is created in 
    cpp in the file vec.h, a wraper is build over it in C in the file stat.h and stat.cpp
    here a python class in the scipt file to wrape up the C api. 
"""

from typing import Any
import pandas as pd
import ctypes

#the glopal cpp dll

file = ctypes.WinDLL(".\\build\\Release\\MyProject.dll")

class Tensor:
    def __init__(self,numrows=None,numColumns=None, python_array=None) -> None:

        if((python_array==None) and (numColumns==None) and (numrows==None)):
            self.pointer = ctypes.c_void_p(0)
            return
        elif((python_array==None) and (numColumns!=None) and (numrows!=None)):
            func = file.createTensorFloat

            func.argtypes = [ctypes.c_size_t,ctypes.c_size_t]
            func.restype = ctypes.c_void_p

            self.pointer = func(numrows,numColumns)
            return
        else:
            func = file.initTensorFloat

            func.argtypes = [ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float)]
            func.restype = ctypes.c_void_p

            c_array = (ctypes.c_float * len(python_array))(*python_array)
            self.pointer = func(numrows,numColumns,c_array)
            return
    

    def getPointer(self):
        return self.pointer
    
    def print(self):
        printf = file.print
        printf.argtypes = [ctypes.c_void_p]
        printf(self.pointer)
        return

    def clean(self):
        func = file.deleteTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func(self.pointer)

    def __add__(self, other ):
        func = file.addTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer, other.getPointer())
        return tf

    def __sub__(self, other ):
        func = file.subTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer, other.getPointer())
        return tf
    
    def __mul__(self, other ):
        func = file.mullTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer, other.getPointer())
        return tf

    def transpose(self):
        func = file.transposeTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer)
        return tf
    
    def inverse(self):
        func = file.inverseTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer)
        return tf

    def setitem(self, i, j, value:float):
        func = file.setElementTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func(self.pointer,i,j,value)
        return 

    def getitem(self, i, j) -> float:
        func = file.getElementTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
        func.restype = ctypes.c_float
        return func(self.pointer,i,j)
         


if(__name__=='__main__'):
     
    data = pd.read_csv("C:\\Users\\mohamed nour\\Downloads\\nba_salary_stats.csv")

    #tf1 = Tensor(python_array = data["fg"].array.tolist())
    #tf2 = Tensor(python_array = data["fga"].array.tolist())
    
    tf1 = Tensor(3,4)
    tf2 = Tensor(3,4)

    tf1.print()
    tf2.print()
    tf3 = tf1.transpose()
    tf3.print()

    tf4 = tf3*tf2
    tf4.print()
    
    print(tf4.getitem(0,0))

    tf1.clean()
    tf2.clean()
    tf3.clean()

   


    


    

    


