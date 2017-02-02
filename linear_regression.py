# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 15:20:57 2017

@author: Rui Hu
"""
import numpy
def predict(alpha,beta,x_i):
    return beta*x_i+alpha

def error(alpha,beta,x_i,y_i):
    return y_i-predict(alpha,beta,x_i)
    
def sum_of_squared_errors(alpha,beta,x,y):
    return sum(error(alpha,beta,x_i,y_i)**2
               for x_i,y_i in zip(x,y))
    
def least_squares_fit(x,y):
    beta=numpy.corrcoef(x,y)[0][1]*numpy.std(y)/numpy.std(x)
    alpha=numpy.mean(y)-beta*numpy.mean(x)
    return alpha,beta
    
def total_sum_of_squares(y):
    return sum((y_i-numpy.mean(y))**2 for y_i in y)
    
def r_squared(alpha,beta,x,y):
    return 1.0-(sum_of_squared_errors(alpha,beta,x,y)/total_sum_of_squares(y))

def squard_error(x_i,y_i,theta):
    alpha,beta=theta
    return error(alpha,beta,x_i,y_i)**2

def squard_error_gradient(x_i,y_i,theta):
    alpha,beta=theta
    return [-2*error(alpha,beta,x_i,y_i),
            -2*error(alpha,beta,x_i,y_i)*x_i]
