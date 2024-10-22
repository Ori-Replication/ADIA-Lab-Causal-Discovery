# Cloud:cloud:
This is an implementation of the following paper:  
[1] Masatoshi, Kobayashi., Kohei, Miyaguchi., Shin, Matsushima. (2022) [Detection of Unobserved Common Cause in Discrete Data Based on the MDL Principle]()


## Requirement
- Python 3.8+
- `numpy`
- `sklearn`

## Install
Clone this repository, then run
```
$ python setup.py install
```
## Usage & Demo
Here is a simple example to use Cloud:

```python
import numpy as np
from cloud import Cloud

# generate data from X causes Y
x = np.random.randint(0, 5, 10000) # 5 cyclic
y = (x + np.random.randint(0, 8, 10000)) % 8 # 8 cyclic
                                             # modulo operation is optional

# pass the data to Cloud
result = Cloud(
    X=x, 
    Y=y,
    n_candidates=4, # select a set of model candidates
    is_print=True # print out inferred causal direction 
    X_ndistinct_vals=5,
    Y_ndistinct_vals=8,
)
```

You should see output like this:

```
Cloud Inference Result:: X ⇒ Y    Δ=2.13
```

`result` is list of tuples (the first element is code-length L(z^n, M), and another is causal model label):

```
[(53272.67451834934, 'to'),
 (53274.80253272231, 'indep'),
 (53277.3410945044, 'gets'),
 (53365.39411832997, 'confounder')]
```

## Run Experimnt
Here is a way to run a experiment in our paper (experiment B). If you want to run other experiments, choose other file names in `experiment/`
```
cd experiments/ && python test_synth.py
```

## Licence
[MIT](https://github.com/Matsushima-lab/Cloud/blob/main/LICENSE)

## Reference
[kbudhath](https://github.molgen.mpg.de/EDA/cisc)

Our implementation is greatly inspired by his.
