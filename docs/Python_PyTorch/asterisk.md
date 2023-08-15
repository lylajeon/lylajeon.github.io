---
title: Use of Asterisk in Python
layout: default
parent: Python, PyTorch
nav_order: 2
---
date: 2023-03-23

Asterisk symbol has various usages and in this article, I would like to review these various usages. This can be divided by [usage related to mathematics](https://lylajeon.github.io/docs/Python_PyTorch/asterisk/#related-to-mathematics) and [usage related to iterables](https://lylajeon.github.io/docs/Python_PyTorch/asterisk/#related-to-iterables). This article is written with the reference of [Yong Cui's Understand the Versatility of asterisks in python](https://betterprogramming.pub/understand-the-versatility-of-asterisks-in-python-know-8-use-cases-722bff20e84c).


## Related to mathematics
1. multiplication
```python
>>> 2*3
6
```
2. power operator 
```python
>>> 2**3
8
```

## Related to iterables 
1. **to capture all in iterable unpacking** <br/>
   This is also called *starred expression*. <br/>
   The usage of this is putting an * operator in front of a variable. <br/>
   Starred expression can be used to capture multiple items when an iterable such as tuple, list, or string is being unpacked. 
```python
data_tuple = (1,2,3)
data_list = [1,2,3]
a, *b = data_tuple # a = 1, b = [2, 3]
c, *d = data_list # c = 1, d = [2, 3]
a, *b, *c = data_tuple # wrong expression (only one starred expression allowed)
```
1. **to unpack iterables for creating variables** <br/>
   Asterisk can also be used to unpack an existing variable.<br/>
```python
existing_items = [1, 2, 3]
new_items = [*existing_items, 4, 5, 6] # [1, 2, 3, 4, 5, 6]
```
1. **to represent varied number of positional arguments in defining functions** <br/>
   When defining functions, there are two types of arguments: positional and keyword. Keyword arguments are those that are specified by identifiers and positional arguments are specified by their positions. <br/>
   Varied number of arguments are processed as a tuple object by the function. <br/>
   Also, the thing to note is that varied number of arguments should be defined after other positional arguments are defined.
```python
def rounded_sum(*numbers):
    total = 0
    for number in numbers:
        total += round(number)
    print(f"Received {numbers} -> rounded sum: {total}")
    return total
>>> rounded_sum(2.4) 
Received (2.4,) -> rounded sum: 2
>>> rounded_sum(2.4, 3.5, 4.6)
Received (2.4, 3.5, 4.6) -> rounded sum: 11
```
1. **to unpack dictionary**<br/>
   One asterisk in front of a dictionary variable can unpack keys of the dictionary. <br/>
   Two asterisks in front of a dictionary variable can unpack both keys and values of the dictionary.
```python
def foo1(a, b, c): 
    print(a, b, c)
science_scores = {"math": 90, "physics": 95, "chemistry": 92}
>>> foo1(*science_scores) 
math physics chemistry
```
```python
science_scores = {"math": 90, "physics": 95, "chemistry": 92}
art_scores = {"english": 93, "spanish": 94}
combined_scores = {**science_scores, **art_scores} # {'math': 90, 'physics': 95, 'chemistry': 92, 'english': 93, 'spanish': 94}
```
1. **to define varied number of keyword arguments** <br/>
   The convention is to use `**kwargs` to denote such a feature. <br/>
   You don’t know what keyword parameters that the function will take.
```python
def send_info_to_server(**kwargs):
    print(f"Send the info: {kwargs}")
    # do something to prepare the information
>>> send_info_to_server(postId="abc", userId="user")
Send the info: {'postId': 'abc', 'userId': 'user'}
>>> send_info_to_server(like=True, status="success")
Send the info: {'like': True, 'status': 'success'}
```
1. **to define keyword-only arguments**<br/>
   These arguments can only be specified using keywords.
```python
def explicit_multiply(number, *, multiplier=1):
    return number * multiplier
>>> explicit_multiply(5, multiplier=4)
20
>>> explicit_multiply(5, 4)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
TypeError: explicit_multiply() takes 1 positional argument but 2 were given
```
   