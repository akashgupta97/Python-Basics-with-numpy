
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


import numpy as np  # this means you can access numpy functions by writing np.function() instead of numpy.function()


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s


# In[19]:

x = np.array([1, 2, 3])
sigmoid(x)

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.

    Arguments:
    x -- A scalar or numpy array
    Return:
    ds -- Your computed gradient.
    """


    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)


    return ds


# In[21]:

x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """


    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

    return v


# In[25]:

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[0.67826139, 0.29380381],
                   [0.90714982, 0.52835647],
                   [0.4215251, 0.45017551]],

                  [[0.92814219, 0.96677647],
                   [0.85304703, 0.52351845],
                   [0.19981397, 0.27417313]],

                  [[0.60659855, 0.00533165],
                   [0.10820313, 0.49978937],
                   [0.34144279, 0.94630077]]])

print("image2vector(image) = " + str(image2vector(image)))


# **Expected Output**:
#
#
# <table style="width:100%">
#      <tr>
#        <td> **image2vector(image)** </td>
#        <td> [[ 0.67826139]
#  [ 0.29380381]
#  [ 0.90714982]
#  [ 0.52835647]
#  [ 0.4215251 ]
#  [ 0.45017551]
#  [ 0.92814219]
#  [ 0.96677647]
#  [ 0.85304703]
#  [ 0.52351845]
#  [ 0.19981397]
#  [ 0.27417313]
#  [ 0.60659855]
#  [ 0.00533165]
#  [ 0.10820313]
#  [ 0.49978937]
#  [ 0.34144279]
#  [ 0.94630077]]</td>
#      </tr>
#
#
# </table>

# ### 1.4 - Normalizing rows
#
# Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to $ \frac{x}{\| x\|} $ (dividing each row vector of x by its norm).
#
# For example, if $$x =
# \begin{bmatrix}
#     0 & 3 & 4 \\
#     2 & 6 & 4 \\
# \end{bmatrix}\tag{3}$$ then $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
#     5 \\
#     \sqrt{56} \\
# \end{bmatrix}\tag{4} $$and        $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
#     0 & \frac{3}{5} & \frac{4}{5} \\
#     \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
# \end{bmatrix}\tag{5}$$

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix.
    """

    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    # Divide x by its norm.
    x = x / x_norm


    return x

