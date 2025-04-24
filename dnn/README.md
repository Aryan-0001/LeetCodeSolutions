# Experiment 1
## Aim: 
- To implement different activation functions commonly used in neural networks using python
## Theory:
- Neural networks rely on activation functions to introduce non-linearity, enabling them to learn complex patterns. Common activation functions include Binary Step, which outputs 0 or 1 based on a threshold, and Linear, which scales inputs directly. Sigmoid maps values between 0 and 1, making it useful for probabilities, while Tanh normalizes between -1 and 1. ReLU (Rectified Linear Unit) sets negative inputs to zero, enhancing gradient flow, whereas Leaky ReLU allows small negative values to avoid dead neurons. Lastly, Softmax converts logits into probabilities, commonly used in classification tasks. Each function plays a crucial role in optimizing neural network performance.

## Program:
```python []
# Binary Step
def binary_step(x):
      if  x<0:    return 0
      else:    return 1
print(f"\nBinary Step:\n{binary_step(5)}")
print(binary_step(-1))

# linear function
def linear_function(x):
        return 4*x
print("\nLinear Function:")
print(linear_function(4))
print(linear_function(-2))

#sigmoid function
import numpy as np
def sigmoid_function(x):
    z=(1/(1+np.exp(-x)))
    return z
print("\nSigmoid Function:")
print(sigmoid_function(7))
print(sigmoid_function(-22))

# tan h function
import numpy as np
def tanh_function(x):
    z=(2/(1+np.exp(-2*x)))-1
    return z
print("\nTanh Function:")
print(tanh_function(0.5))
print(tanh_function(-1))

# relu function
def relu_function(x):
    if(x<0): return 0
    else: return x

print("\nRelu Function:")
print(relu_function(7))
print(relu_function(-7))

#leaky relu function
def leaky_relu_function(x):
    if(x<0): return 0.01*x
    else: return x
print("\nLeaky Relu Function:")

print(leaky_relu_function(7))
print(leaky_relu_function(-7))

# softmax function
def softmax_function(x):
    z=np.exp(x)
    z_ = z/z.sum()
    return z_
print("\nSoftmax Function:")
print(softmax_function([0.8,1.2,3.1]))
```
## Output:

---

<br>

<br>

# Experiment 2
## Aim: 
- a) To classify the following data points using perceptron. Consider a step activation function
- b) with the same value of input, weights, and bias find the output using sigmoid activation function.
## Theory:
  - A perceptron is one of the fundamental building blocks of neural networks and is primarily used for binary classification tasks. It consists of multiple input features, each associated with a weight, a bias term, and an activation function that determines the output.
    In part (a), the step activation function (also called the threshold function) is used, which outputs either 0 or 1 depending on whether the weighted sum of inputs exceeds a threshold. This makes it effective for problems with clear decision boundaries but limits its use for complex or non-linearly separable data.
    In part (b), the sigmoid activation function is applied, which outputs values between 0 and 1. Unlike the step function, the sigmoid function is continuous and differentiable, making it more suitable for gradient-based learning methods. It helps in probabilistic interpretation, as the output can be viewed as the likelihood of a particular class.

## Program (a):
```python []
import numpy as np 
# Step activation function (Threshold function)
def step_activation(z):
    return 1 if z >= 0 else 0
# Input dataset
X = np.array([[3, 2], [0, 1], [-2, 1]])
# Weights and bias initialization
w1, w2 = -0.7, 0.3
b = -0.2
# Perceptron classification function
def perceptron_classify(X, w1, w2, b):
    outputs = []
    for x1, x2 in X:
        # Compute the weighted sum
        z = w1 * x1 + w2 * x2 + b
        # Apply activation function
        output = step_activation(z)
        # Store the result
        outputs.append(output)
    return outputs
# Get perceptron outputs
outputs = perceptron_classify(X, w1, w2, b)
# Print results
for i, (x1, x2) in enumerate(X):
    print(f"Input: ({x1}, {x2}), Output: {outputs[i]}")
```
### Output:

---


## Program (b):
```python []
import numpy as np 
# Sigmoid activation function
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))
# Input dataset
X = np.array([[3, 2], [0, 1], [-2, 1]])
# Weights and bias initialization
w1, w2 = -0.7, 0.3
b = -0.2
# Perceptron classification function
def perceptron_classify(X, w1, w2, b):
    outputs = []
    for x1, x2 in X:
        # Compute the weighted sum
        z = w1 * x1 + w2 * x2 + b
        # Apply sigmoid activation function
        output = sigmoid_activation(z)
        # Store the result
        outputs.append(output)
    return outputs
# Get perceptron outputs
outputs = perceptron_classify(X, w1, w2, b)
for i, (x1, x2) in enumerate(X):
    print(f"Input: ({x1}, {x2}), Output: {outputs[i]:.4f}")
```
### Output:
 
---

# Experiment 3
## Aim: 
- To understand vectorization using numpy
## Theory:
- Vectorization is a technique in numerical computing that allows operations to be performed on entire arrays instead of using explicit loops, leading to significant performance improvements. NumPy, a powerful Python library for numerical computing, enables efficient vectorized operations, making computations faster and more memory-efficient.
  In this experiment, non-vectorized and vectorized implementations of forward propagation in a neural network are compared. The non-vectorized approach iterates through input data using nested loops, performing element-wise computations. In contrast, the vectorized approach leverages NumPy's dot product (np.dot) to compute results in a single operation, reducing execution time.
  Vectorization is crucial in machine learning and deep learning, as it speeds up matrix computations involved in training and inference, enabling efficient handling of large datasets.
## Program:
```python []
import numpy as np
# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Non-vectorized forward propagation
def non_vectorized(X, W, b):
    outputs = []
    for i in range(len(X)):
        z = 0
        for j in range(len(W)):
            z += W[j] * X[i][j]
        z += b
        outputs.append(sigmoid(z))
    return outputs
# Vectorized forward propagation
def vectorized(X, W, b):
    Z = np.dot(X, W) + b
    return sigmoid(Z)
X = np.array([[1, 2], 
              [0, 1], 
              [2, 3]])
W = np.array([0.5, -0.3])
b = 0.1
print("Non-vectorized:", [f"{x:.4f}" for x in non_vectorized(X, W, b)])
print("Vectorized:", [f"{x:.4f}" for x in vectorized(X, W, b)])
```

### Output:
 

---


# Experiment 4
## Aim: 
- To Implement AND, OR, NOR, NAND, XOR, and XNOR using ANN
## Theory:
- Logic gates like AND, OR, NOR, NAND, XOR, and XNOR can be implemented using Perceptrons in an Artificial Neural Network (ANN). A perceptron computes the weighted sum of inputs and applies an activation function (unit step function):
- y = f ( W • X + b)
- where WWW is the weight, XXX is the input, and bbb is the bias.
	AND & OR: Single-layer perceptrons with appropriate weights and bias.
	NAND & NOR: AND/OR perceptron followed by a NOT perceptron.
	XOR & XNOR: Require a multi-layer perceptron (combining AND, OR, and NOT) since they are not linearly separable.
- This experiment demonstrates how single-layer perceptrons can implement basic logic functions, while multi-layer networks are needed for more complex operations.

## Program: 
```python []
# AND Gate
# importing Python library 
import numpy as np 
# define Unit Step Function 
def unitStep(v): 
	if v >= 0: 
		return 1
	else: 
		return 0
# design Perceptron Model 
def perceptronModel(x, w, b): 
	v = np.dot(w, x) + b 
	y = unitStep(v) 
	return y 
# AND Logic Function 
# w1 = 1, w2 = 1, b = -1.5 
def AND_logicFunction(x): 
	w = np.array([1, 1]) 
	b = -1.5
	return perceptronModel(x, w, b) 
# testing the Perceptron Model 
test1 = np.array([0, 1]) 
test2 = np.array([1, 1]) 
test3 = np.array([0, 0]) 
test4 = np.array([1, 0]) 
print("AND({}, {}) = {}".format(0, 1, AND_logicFunction(test1))) 
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(test2))) 
print("AND({}, {}) = {}".format(0, 0, AND_logicFunction(test3))) 
print("AND({}, {}) = {}".format(1, 0, AND_logicFunction(test4))) 
```
### Output:

```python [] 
# OR Gate
# importing Python library 
import numpy as np 
# define Unit Step Function 
def unitStep(v): 
	if v >= 0: 
		return 1
	else: 
		return 0
# design Perceptron Model 
def perceptronModel(x, w, b): 
	v = np.dot(w, x) + b 
	y = unitStep(v) 
	return y 
# OR Logic Function 
# w1 = 1, w2 = 1, b = -0.5 
def OR_logicFunction(x): 
	w = np.array([1, 1]) 
	b = -0.5
	return perceptronModel(x, w, b) 
# testing the Perceptron Model 
test1 = np.array([0, 1]) 
test2 = np.array([1, 1]) 
test3 = np.array([0, 0]) 
test4 = np.array([1, 0]) 
print("OR({}, {}) = {}".format(0, 1, OR_logicFunction(test1))) 
print("OR({}, {}) = {}".format(1, 1, OR_logicFunction(test2))) 
print("OR({}, {}) = {}".format(0, 0, OR_logicFunction(test3))) 
print("OR({}, {}) = {}".format(1, 0, OR_logicFunction(test4))) 
```
Output: 
 
```python []
# NOR Gate
# importing Python library 
import numpy as np 
# define Unit Step Function 
def unitStep(v): 
	if v >= 0: 
		return 1
	else: 
		return 0
# design Perceptron Model 
def perceptronModel(x, w, b): 
	v = np.dot(w, x) + b 
	y = unitStep(v) 
	return y 

# NOT Logic Function 
# wNOT = -1, bNOT = 0.5 
def NOT_logicFunction(x): 
	wNOT = -1
	bNOT = 0.5
	return perceptronModel(x, wNOT, bNOT) 

# OR Logic Function 
# w1 = 1, w2 = 1, bOR = -0.5 
def OR_logicFunction(x): 
	w = np.array([1, 1]) 
	bOR = -0.5
	return perceptronModel(x, w, bOR) 

# NOR Logic Function 
# with OR and NOT 
# function calls in sequence 
def NOR_logicFunction(x): 
	output_OR = OR_logicFunction(x) 
	output_NOT = NOT_logicFunction(output_OR) 
	return output_NOT 

# testing the Perceptron Model 
test1 = np.array([0, 1]) 
test2 = np.array([1, 1]) 
test3 = np.array([0, 0]) 
test4 = np.array([1, 0]) 

print("NOR({}, {}) = {}".format(0, 1, NOR_logicFunction(test1))) 
print("NOR({}, {}) = {}".format(1, 1, NOR_logicFunction(test2))) 
print("NOR({}, {}) = {}".format(0, 0, NOR_logicFunction(test3))) 
print("NOR({}, {}) = {}".format(1, 0, NOR_logicFunction(test4))) 
```
### Output:

 ```python []
# NAND Gate
# importing Python library 
import numpy as np 

# define Unit Step Function 
def unitStep(v): 
	if v >= 0: 
		return 1
	else: 
		return 0

# design Perceptron Model 
def perceptronModel(x, w, b): 
	v = np.dot(w, x) + b 
	y = unitStep(v) 
	return y 

# NOT Logic Function 
# wNOT = -1, bNOT = 0.5 
def NOT_logicFunction(x): 
	wNOT = -1
	bNOT = 0.5
	return perceptronModel(x, wNOT, bNOT) 

# AND Logic Function 
# w1 = 1, w2 = 1, bAND = -1.5 
def AND_logicFunction(x): 
	w = np.array([1, 1]) 
	bAND = -1.5
	return perceptronModel(x, w, bAND) 

# NAND Logic Function 
# with AND and NOT 
# function calls in sequence 
def NAND_logicFunction(x): 
	output_AND = AND_logicFunction(x) 
	output_NOT = NOT_logicFunction(output_AND) 
	return output_NOT 

# testing the Perceptron Model 
test1 = np.array([0, 1]) 
test2 = np.array([1, 1]) 
test3 = np.array([0, 0]) 
test4 = np.array([1, 0]) 

print("NAND({}, {}) = {}".format(0, 1, NAND_logicFunction(test1))) 
print("NAND({}, {}) = {}".format(1, 1, NAND_logicFunction(test2))) 
print("NAND({}, {}) = {}".format(0, 0, NAND_logicFunction(test3))) 
print("NAND({}, {}) = {}".format(1, 0, NAND_logicFunction(test4))) 
```
## Output:

```python []
# XOR Gate
# importing Python library
import numpy as np

# define Unit Step Function
def unitStep(v):
	if v >= 0:
		return 1
	else:
		return 0

# design Perceptron Model
def perceptronModel(x, w, b):
	v = np.dot(w, x) + b
	y = unitStep(v)
	return y

# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
	wNOT = -1
	bNOT = 0.5
	return perceptronModel(x, wNOT, bNOT)

# AND Logic Function
# here w1 = wAND1 = 1, 
# w2 = wAND2 = 1, bAND = -1.5
def AND_logicFunction(x):
	w = np.array([1, 1])
	bAND = -1.5
	return perceptronModel(x, w, bAND)

# OR Logic Function
# w1 = 1, w2 = 1, bOR = -0.5
def OR_logicFunction(x):
	w = np.array([1, 1])
	bOR = -0.5
	return perceptronModel(x, w, bOR)

# XOR Logic Function
# with AND, OR and NOT 
# function calls in sequence
def XOR_logicFunction(x):
	y1 = AND_logicFunction(x)
	y2 = OR_logicFunction(x)
	y3 = NOT_logicFunction(y1)
	final_x = np.array([y2, y3])
	finalOutput = AND_logicFunction(final_x)
	return finalOutput

# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("XOR({}, {}) = {}".format(0, 1, XOR_logicFunction(test1)))
print("XOR({}, {}) = {}".format(1, 1, XOR_logicFunction(test2)))
print("XOR({}, {}) = {}".format(0, 0, XOR_logicFunction(test3)))
print("XOR({}, {}) = {}".format(1, 0, XOR_logicFunction(test4)))
```
## Output:

```python []
# XNOR Gate
# importing Python library
import numpy as np

# define Unit Step Function
def unitStep(v):
	if v >= 0:
		return 1
	else:
		return 0

# design Perceptron Model
def perceptronModel(x, w, b):
	v = np.dot(w, x) + b
	y = unitStep(v)
	return y

# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
	wNOT = -1
	bNOT = 0.5
	return perceptronModel(x, wNOT, bNOT)

# AND Logic Function
# w1 = 1, w2 = 1, bAND = -1.5
def AND_logicFunction(x):
	w = np.array([1, 1])
	bAND = -1.5
	return perceptronModel(x, w, bAND)

# OR Logic Function
# here w1 = wOR1 = 1, 
# w2 = wOR2 = 1, bOR = -0.5
def OR_logicFunction(x):
	w = np.array([1, 1])
	bOR = -0.5
	return perceptronModel(x, w, bOR)

# XNOR Logic Function
# with AND, OR and NOT 
# function calls in sequence
def XNOR_logicFunction(x):
	y1 = OR_logicFunction(x)
	y2 = AND_logicFunction(x)
	y3 = NOT_logicFunction(y1)
	final_x = np.array([y2, y3])
	finalOutput = OR_logicFunction(final_x)
	return finalOutput

# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("XNOR({}, {}) = {}".format(0, 1, XNOR_logicFunction(test1)))
print("XNOR({}, {}) = {}".format(1, 1, XNOR_logicFunction(test2)))
print("XNOR({}, {}) = {}".format(0, 0, XNOR_logicFunction(test3)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_logicFunction(test4)))
```

## Output:

 ---
 ---
 
# Experiment 5
# Aim: 
- To implement the linear regression model with one variable for housing price prediction
# Theory:
- Linear regression is a statistical method used to model the relationship between a dependent variable (in this case, housing price) and an independent variable (house size in square feet). The model assumes that this relationship can be approximated by a straight line. The goal is to find the optimal parameters, which include the weight (slope) and bias (intercept), to best fit the data.
  In linear regression, the price of a house is predicted using the equation:
  				Price = (w . size) + b
  where:
  	www is the weight or slope of the line, representing the rate at which the price changes as the size increases.
  	bbb is the bias or intercept, representing the predicted price when the house size is zero.
## Program:
```python []
import numpy as np
import matplotlib.pyplot as plt
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
w,b = 200,100
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```
### Output:
 

 
 
---
---

<br>

# Experiment 6
## Aim: 
- For the same housing price example, automate the process of optimizing 𝑤 and 𝑏 using gradient descent
## Theory:
- This experiment automates the optimization of the parameters www (weight) and bbb (bias) in a linear regression model using Gradient Descent. The goal is to minimize the cost function (Mean Squared Error) which measures the difference between the predicted and actual values.
  Gradient Descent works by iteratively adjusting www and bbb to reduce the cost. The key steps are:
	Initialization: Start with initial guesses for www and bbb.
	Compute gradients: Calculate how much the cost function changes with respect to www and bbb.
	Update parameters: Adjust www and bbb using the gradients and a learning rate α\alphaα.
	Repeat: Continue until the cost function converges.
  By minimizing the cost function, Gradient Descent finds the optimal values of www and bbb that best fit the model to the data.
## Program:
```python []
import math, copy
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
# Load our data set
x_train = np.array([1.0, 2.0]) #features
y_train = np.array([300.0, 500.0]) #target value
# Function to calculate the cost
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

for i in range(num_iters):
    # Calculate the gradient and update the parameters using gradient_function
    dj_dw, dj_db = gradient_function(x, y, w, b)

    # Update Parameters using equation (3) above
    b = b - alpha * dj_db
    w = w - alpha * dj_dw

    # Save cost J at each iteration
    if i < 100000:  # prevent resource exhaustion
        J_history.append(cost_function(x, y, w, b))
        p_history.append([w, b])

    # Print cost every at intervals 10 times or as many iterations if < 10
    if i % math.ceil(num_iters / 10) == 0:
        print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
              f"dJ_dw: {dj_dw: 0.3e}  dJ_db: {dj_db: 0.3e}",
              f"w: {w: 0.3e} b: {b: 0.5e}")

return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)"); ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')                 ; ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step') ; ax2.set_xlabel('iteration step')
import math, copy
import numpy as np

import matplotlib.pyplot as plt

#plt.style.use('./deeplearning.mplstyle')

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Load our data set
x_train = np.array([1.0, 2.0]) #features
y_train = np.array([300.0, 500.0]) #target value

# Function to calculate the cost
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dJ_dw: {dj_dw: 0.3e}  dJ_db: {dj_db: 0.3e}",
                  f"w: {w: 0.3e} b: {b: 0.5e}")

    return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)"); ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')                 ; ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step') ; ax2.set_xlabel('iteration step')
plt.show()
```

### Output:

---
---

# Experiment 7
## Aim: 
- To implement and update the weights for the network shown. Considering target output at 0.5 and learning rate as 1. ![image](https://github.com/user-attachments/assets/52ee4a70-ca61-4558-993f-e9796344cd50)

 
## Theory:
  - In this experiment, a Neural Network is implemented to solve a simple binary classification problem, specifically learning the XOR function. The network consists of an input layer, a hidden layer, and an output layer.
    The key components are:
      Feedforward Process: Data is passed through the network, where each layer's output is computed using the weights, biases, and the sigmoid activation function.
      Backpropagation: The error between the predicted and target outputs is propagated backward through the network, and the weights and biases are updated using the gradient descent method. The learning rate controls the step size during updates.
      Training: The network is trained over multiple epochs, during which the weights are continually updated to minimize the error between predicted and target values.
    The goal is to adjust the weights using gradient descent to minimize the loss, with a target output of 0.5 and a learning rate of 1. The network learns to approximate the XOR function over time through iterations.
## Program:
```python []
import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def feedforward(self, X):
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)
        return self.predicted_output
    def backward(self, X, y, learning_rate):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)
output = nn.feedforward(X)
print("Predictions after training:")
print(output)
```
### Output:
  
# Experiment 8
## Aim: 
 - To Implement ANN with tensorflow on:
    - on power plant data set
    - on churn data set 
## Theory:
- In this experiment, Artificial Neural Networks (ANNs) are implemented using TensorFlow and Keras for two different tasks:
	Power Plant Dataset (Regression):
	The dataset predicts the net electrical energy output of a power plant based on various environmental and operational features like temperature, pressure, and humidity.
	The model uses a feedforward neural network with multiple hidden layers and the ReLU activation function. The loss function is mean squared error (MSE), which is suitable for regression problems.
	The network is trained using the Adam optimizer, and evaluation metrics such as mean squared logarithmic error are used.
	Churn Dataset (Classification):
	The dataset is used to predict whether a customer will churn (leave) or stay based on features like age, balance, and credit score.
	The network uses sigmoid activation at the output layer to perform binary classification. The binary cross-entropy loss function is used to evaluate model performance.
	The model is evaluated based on accuracy, confusion matrix, and the ROC curve, which helps assess the model's classification ability.
- Both implementations involve preprocessing the data (normalization, encoding categorical features) and splitting the data into training and testing sets for evaluation.
## Program:
a). Power plant data set
```python []
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from math import sqrt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
# Load dataset
Powerplant_data = pd.read_excel('/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/Folds5x2_pp.xlsx')
# Display basic information
print(Powerplant_data.head(5))
print(Powerplant_data.columns)
print("Dataset Shape:", Powerplant_data.shape)
print(Powerplant_data.info())
print("Missing Values:")
print(Powerplant_data.isna().sum())
print("Unique Values per Column:")
print(Powerplant_data.nunique())
# Splitting data into features and target variable
X = Powerplant_data.iloc[:, :-1].values
y = Powerplant_data.iloc[:, -1].values
# Splitting into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
# Print dataset shapes
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print("Shape of y_val:", y_val.shape)
# Build ANN model
classifier = Sequential()
# Adding layers
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=4))  # Input layer
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))  # Hidden layer 1
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))  # Hidden layer 2
classifier.add(Dense(units=1, kernel_initializer='uniform'))  # Output layer
# Compile the model
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanSquaredLogarithmicError'])
# Train the model
model = classifier.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val), shuffle=True)
# Predict on test data
y_pred = classifier.predict(X_test)
# Display predictions
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# Evaluate model
mae_no = sklearn.metrics.mean_absolute_error(y_test, classifier.predict(X_test))
mse_no = sklearn.metrics.mean_squared_error(y_test, classifier.predict(X_test))
rms = sqrt(sklearn.metrics.mean_squared_error(y_test, classifier.predict(X_test)))
# Print evaluation metrics
print('Mean Absolute Error     :', mae_no)
print('Mean Square Error       :', mse_no)
print('Root Mean Square Error  :', rms)
# Enable inline plotting
%matplotlib inline
```
### Output:
 
 
     
 

b) churn data set
```python []
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense

# Load and preprocess data
churn_data = pd.read_csv('/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/Churn_Modelling.csv', delimiter=',')
churn_data = churn_data.set_index('RowNumber')
churn_data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
churn_data[['Geography', 'Gender']] = churn_data[['Geography', 'Gender']].apply(le.fit_transform)

# Split features and target
X = churn_data.drop(['Exited'], axis=1)
y = churn_data['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shapes
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build ANN model
classifier = Sequential()
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=10))
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile and train model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)

# Evaluate on training set
train_score, train_acc = classifier.evaluate(X_train, y_train, batch_size=10)
print('Train score:', train_score)
print('Train accuracy:', train_acc)

# Predict and evaluate on test set
y_pred = classifier.predict(X_test)
y_pred_binary = (y_pred > 0.5)

print('*' * 20)
test_score, test_acc = classifier.evaluate(X_test, y_test, batch_size=10)
print('Test score:', test_score)
print('Test accuracy:', test_acc)

# Confusion Matrix
target_names = ['Retained', 'Closed']
cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cm), annot=True, xticklabels=target_names, yticklabels=target_names, 
            cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=target_names))

# ROC Curve
y_pred_proba = classifier.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label=f'AUC (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(loc="lower right")
plt.title('ROC Curve')
plt.show()
# AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Area under ROC curve: {auc_score:.4f}")
```

### Output:
 
 
# Experiment 9
## Aim: 
  - Implement using CNN: For the 3x3 image and 2x2 filter, assuming a stride of 1 and no padding, find the output feature map. Also write a tensorflow/python code to implement this
    <br>
    ![image](https://github.com/user-attachments/assets/8b88be72-b8f9-4fa8-b3ad-4d10c47dd8d5)
    <br>

    Consider a 32x32 grayscale image as input, apply the following layers:
    Convolutional layer: 6 filters (5x5), stride = 1, no padding
    Max pooling layer: pool size (2x2), stride = 2
    Another convolutional layer: 16 filters (3x3), stride = 1, no padding
    Max pooling layer: pool size (2x2), stride = 2
    Find the dimensions of the output image at the end of each layer with the help of tensorflow/python codes.
## Theory:
- In this experiment, we implement a Convolutional Neural Network (CNN) using TensorFlow for two tasks:
	3x3 Image and 2x2 Filter (Manual Calculation):
	We are given a 3x3 input image and a 2x2 filter. The convolution operation is performed by sliding the filter over the image with a stride of 1 and no padding.
	The output is calculated by performing element-wise multiplication of the filter and the image region and then summing the results for each position. This results in a 2x2 output feature map.
	CNN with 32x32 Grayscale Image (Using TensorFlow):
	The model consists of multiple layers: 
	Convolutional layer with 6 filters of size 5x5, stride 1, and no padding.
	Max pooling layer with a 2x2 pool size and a stride of 2.
	Another Convolutional layer with 16 filters of size 3x3, stride 1, and no padding.
	Another Max pooling layer.
	After each layer, the dimensions of the output image are calculated using the formula based on the input size, filter size, stride, and padding. The dimensions of the output after each layer are printed using TensorFlow.

## Program a): 
```python []
import tensorflow as tf
import numpy as np
# Define 3x3 input image
input_image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
my_filter = np.array([
    [1, -1],
    [-1, 1]
])
input_image.shape
my_filter.shape

input_image_height, input_image_width = input_image.shape
output_height, output_width = input_image_height, input_image_width
filter_height, filter_width = my_filter.shape
output = np.zeros((output_height, output_width))  # Initialize output matrix
for i in range(output_height):  # Sliding over rows
    for j in range(output_width):  # Sliding over columns
        region = input_image[i:i+filter_height, j:j+filter_width]  # Extracting 2x2 region
        output[i, j] = np.sum(region * my_filter)  # Element-wise multiplication & sum
print("Output Feature Map:")
print(output)
```

TensorFlow Implementation
```python []
import tensorflow as tf
import numpy as np

# Define 3x3 input image
input_image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32).reshape(1, 3, 3, 1)  # Adding batch and channel dimensions

# Define 2x2 filter (kernel)
kernel = np.array([
    [1, -1],
    [-1, 1]
], dtype=np.float32).reshape(2, 2, 1, 1)  # (Filter height, Filter width, Input channels, Output channels)

# Apply convolution using TensorFlow
conv_layer = tf.nn.conv2d(input_image, kernel, strides=[1, 1, 1, 1], padding="VALID")
# Run the TensorFlow session and print the output
print("Output Feature Map:")
print(conv_layer.numpy().squeeze())  # Remove unnecessary dimensions
```
### Output:
   
 
## Program(b)
```python []
import tensorflow as tf
# Define the input shape (batch size ignored, grayscale image 32x32)
input_shape = (32, 32, 1)  # Grayscale image
# Create a sequential model
model = tf.keras.Sequential()
# First Convolutional Layer (6 filters, 5x5 kernel, stride 1, no padding)
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=1, padding='valid', activation='relu', input_shape=input_shape))
# First Max Pooling Layer (2x2 pool, stride 2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
# Second Convolutional Layer (16 filters, 3x3 kernel, stride 1, no padding)
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
# Second Max Pooling Layer (2x2 pool, stride 2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
# Print model summary to verify dimensions
model.summary()
```

### Output:


# Experiment 10
## Aim: 
- Implement RNN using tensorflow with input values x = [1, 2, 3], input-hidden layer weights Wh = 0.5, Vh = 0.3, output layer weights Wy = 0.7, bias terms bh=0.1, by = 0.2, h0 = 0. Consider tan h as activation function. 
## Theory:
- This experiment demonstrates the implementation of a Recurrent Neural Network (RNN) using TensorFlow. In the first part, the model is used without training. A simple RNN with a tanh activation function processes an input sequence x = [1, 2, 3] using predefined weights and biases to compute the output at each time step. The weights (Wh = 0.5, Vh = 0.3, Wy = 0.7) and bias terms (bh = 0.1, by = 0.2) are manually assigned to the RNN layer. The forward pass computes the output at each time step by applying the tanh activation function.
  In the second part, the model is trained with the input sequence x_train = [1, 2, 3] and corresponding target outputs y_train = [0.5, 0.7, 0.9]. The model is trained using backpropagation with the mean squared error (MSE) loss function, optimized by the Adam optimizer. The training process involves 500 epochs, and after training, the network can predict the output sequence at each time step based on the learned weights.
  This experiment helps in understanding the basic working of RNNs, showing both the forward pass with fixed weights and the training process to adjust weights for improved predictions.
## Program:
Without training
```python []
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
x_sequence = np.array([[1, 2, 3]])  # Shape: (batch_size=1, time_steps=3, features=1)
x_sequence = x_sequence.reshape((1, 3, 1))  # Reshape for RNN input (batch_size, time_steps, features)
# Define the RNN model
model = Sequential([
    SimpleRNN(units=1, activation='tanh', return_sequences=True, input_shape=(3, 1)),  # RNN layer
    Dense(1)  # Output layer
])
W_h = np.array([[0.5]])  # Input weight
U_h = np.array([[0.3]])  # Recurrent weight
b_h = np.array([0.1])  # Bias for hidden state
W_y = np.array([[0.7]])  # Output weight
b_y = np.array([0.2])  # Bias for output
# Assign the weights to the RNN layer
model.layers[0].set_weights([W_h, U_h, b_h])
model.layers[1].set_weights([W_y, b_y])
# Forward pass (prediction)
output = model.predict(x_sequence)
print("Outputs at each time step:")
for t, y_t in enumerate(output[0]):
    print(f"Time step {t+1}: y={y_t[0]:.3f}")
```
### Output:
 
With training
```python []
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
# Define input sequence and target outputs
x_train = np.array([[1, 2, 3]])  # Input sequence (batch_size=1, time_steps=3, features=1)
y_train = np.array([[0.5, 0.7, 0.9]])  # Target outputs
# Reshape for RNN input (batch_size, time_steps, features)
x_train = x_train.reshape((1, 3, 1))
y_train = y_train.reshape((1, 3, 1))
# Define the RNN model
model = Sequential([
    SimpleRNN(units=1, activation='tanh', return_sequences=True, input_shape=(3, 1)),  # RNN layer
    Dense(1)  # Output layer
])
# Compile the model with loss function and optimizer
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
# Train the model using backpropagation
print("Training the model...")
model.fit(x_train, y_train, epochs=500, verbose=0)  # Train for 500 epochs
# Forward pass (prediction after training)
output = model.predict(x_train)
# Print final predictions
print("Outputs after training:")
for t, y_t in enumerate(output[0]):
    print(f"Time step {t+1}: y={y_t[0]:.3f}")

```
Output:
 

