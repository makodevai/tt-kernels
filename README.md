# tt-kernels
Generated kernels from Mako for TT


## Generated kernels

Unary
* Trigonometric
    * atanh
    * acosh
    * asinh
* Activations
    * softsign
    * hardsigmoid
* Specialized functions
    * celu - needs review -- abs_error = 0.0117 for 64 x 64 input shape. For small (1x1, 1x2,...) error is 0


## How to use

Example
```
PYTHONPATH=$PWD python3 '/home/george/tt-kernels/kernels/binary/hypot/fused/test.py'
PYTHONPATH=$PWD python3 '/home/george/tt-kernels/kernels/binary/hypot/fused/host.py'
```