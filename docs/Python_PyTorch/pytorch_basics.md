---
title: Pytorch Basics (1)
layout: default
parent: Python, PyTorch
# nav_order: 2
---
date: 2022-09-26

# Introduction to PyTorch

## TensorFlow vs PyTorch

Leaders of Deep Learning Framework is TensorFlow (Google) and **PyTorch** (Facebook). 

TensorFlow and PyTorch are mainly different in computational graph used in each framework. TensorFlow uses static graph, and PyTorch uses dynamic computation graphs. 

### Static vs Dynamic Graph

Static graph uses Define and Run method. It defines the graph first, and then feed the data at the time of execution. 

Dynamic computation graph (DCG) uses Define by Run method. It generates the graph with the execution. 

Static graph builds graph once, then run many times. In contrast, in dynamic graph, each forward pass defines a new graph. 

DCG is better for *debugging* than static graph. 

### Why PyTorch

- The benefit of Define by Run, which is easier to debug and use pythonic code. 
- Supports GPU and has good API and community 
- Easy to use 

### PyTorch 

Numpy + Autograd + Function

- Express array as tensor object that has Numpy structure. 
- Supports deep learning computation by Autograd
- Supports functions and models of various forms of deep learning 

---

# PyTorch Basics 

## Tensor 

PyTorch class that represents multi-dimensional arrays, which is similar to ndarray in numpy and the function that generates tensor is also similar. 

Making new tensor

```python
# numpy - ndarray
import numpy as np
n_array = np.arange(10).reshape(2,5)
print(n_array)
print(f"ndim: {n_array.ndim} shape: {n_array.shape}")

# pytorch - tensor
import torch 
t_array = torch.FloatTensor(n_array)
print(t_array)
print(f"ndim: {t_array.ndim} shape: {t_array.shape}")
```



squeeze vs unsqueeze

numpy operation 

reshape 대신 view 써라 

mm 과 dot, matmul 차이 