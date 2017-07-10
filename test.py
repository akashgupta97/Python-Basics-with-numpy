
test = "Hello World"

print("test: " + test)

# **Expected output**:
# test: Hello World

# <font color='blue'>

# ## 1 - Building basic functions with numpy ##
#
# Numpy is the main package for scientific computing in Python. It is maintained by a large community (www.numpy.org). In this exercise you will learn several key numpy functions such as np.exp, np.log, and np.reshape. You will need to know how to use these functions for future assignments.
#
# ### 1.1 - sigmoid function, np.exp() ###
#
# Before using np.exp(), you will use math.exp() to implement the sigmoid function. You will then see why np.exp() is preferable to math.exp().
#
#
# **Reminder**:
# $sigmoid(x) = \frac{1}{1+e^{-x}}$ is sometimes also known as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.
#
# <img src="images/Sigmoid.png" style="width:500px;height:228px;">
#
# To refer to a function belonging to a specific package you could call it using package_name.function(). Run the code below to see an example with math.exp().

# In[13]:


import math


def basic_sigmoid(x):
    """
    Compute sigmoid of x.
    Arguments:
    x -- A scalar
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + math.exp(-x))

    return s


# In[14]:

basic_sigmoid(3)

# **Expected Output**:
# <table style = "width:40%">
#     <tr>
#     <td>** basic_sigmoid(3) **</td>
#         <td>0.9525741268224334 </td>
#     </tr>
#
# </table>


# In[15]:

### One reason why we use "numpy" instead of "math" in Deep Learning ###
x = [1, 2, 3]
###basic_sigmoid(x)  # you will see this give an error when you run it, because x is a vector.

# In fact, if $ x = (x_1, x_2, ..., x_n)$ is a row vector then $np.exp(x)$ will apply the exponential function to every element of x. The output will thus be: $np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})$

# In[16]:

import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x))  # result is (exp(1), exp(2), exp(3))

# Furthermore, if x is a vector, then a Python operation such as $s = x + 3$ or $s = \frac{1}{x}$ will output s as a vector of the same size as x.

# In[17]:

# example of vector operation
x = np.array([1, 2, 3])
print(x + 3)

