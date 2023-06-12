# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:30:59 2023

@author: sinar
"""

import numpy as np

#assign input values
input_value = np.array([[0,0],[0,1],[1,1],[1,0]])
input_value.shape
print(input_value)

#assign output values
output = np.array([0,1,1,0])
output = output.reshape(4,1)
print(output.shape)

#assign weights
weights = np.array([[0.1],[0.2]])
print(weights)

#add bias
bias = 0.3

#activation function
def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

#derivative of sigmoid function
def der(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))


#updating weights
for epochs in range(1000):
    input_arr = input_value
    
    weighted_sum = np.dot(input_arr, weights) + bias
    first_output = sigmoid_func(weighted_sum)
    
    error = first_output - output
    total_error = np.square(np.subtract(first_output, output)).mean() #mean square error
    #print(total_error)
    
    first_der = error
    second_der = der(first_output)
    derivative = first_der * second_der
    
    t_input = input_value.T
    final_derivative = np.dot(t_input, derivative)
    
    #update weights
    weights = weights - 0.05 * final_derivative

    #update bias
    for i in derivative:
        bias = bias - 0.05 * i
    
print(weights)
print(bias)

#predictions
pred = np.array([0,1])
result = np.dot(pred, weights) + bias
res = sigmoid_func(result)
print (res)    
    
    
    