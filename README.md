
## Control design based on linear matrix inequalities for integration of distributed energy resouces in microgrid systems.

### Introduction
This algorithm solves the primary control synthesis problem for integration of DER units in microgrid systems. The optimization problem is formulated with linear matrix inequalities and it is solved using cvxpy as interpreter with the solver CVXOPT.   

### Dependencies

### Example
In `example.py` an example of a DER unit of 2[MVA] is provided. The nominal values and the circuit parameters are provided in `--SYSTEM DEFINITION--` section. In `main_LMI.py`, local functions are difined, which are called from `example.py`. To execute the example, just run:

```
python example.py
```
The algorithm converges in few seconds. If desired, the nominal values and circuit parameters can be modified for a different DER unit with other sampling time. If that is the case, and the control design problem result infeasible, 
the authors recommend to adjust the `gamma_(j)` values.


