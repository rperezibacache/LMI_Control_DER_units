
## Control design based on linear matrix inequalities for integration of distributed energy resouces in microgrid systems.

### Introduction
This algorithm solves the primary control synthesis problem for integration of DER units in microgrid systems. The optimization problem is formulated with linear matrix inequalities and it is solved using cvxpy as interpreter with the solver CVXOPT.   

### Dependencies

### Example
In *example.py* an example of a DER unit of 2[MVA] is provided. The nominal values and the parameters are provide in *SYSTEM DEFINITION* section. In *main_LMI.py* a group of functions definitions are provided, which are called from *example.py*. To execute the example, just run:

```
python example.py
```




