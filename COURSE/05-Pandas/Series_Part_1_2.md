```python
import numpy as np
```


```python
import pandas as pd
```


```python
# help(pd.Series)
```


```python
myindex = ['USA', 'Canada', 'Mexico']
```


```python
mydata = [1776, 1867, 1821]
```


```python
myser = pd.Series(data=mydata, index=myindex)
print(myser)
type(myser)
```

    USA       1776
    Canada    1867
    Mexico    1821
    dtype: int64





    pandas.core.series.Series




```python
print(myser[0])
print(myser['USA'])
```

    1776
    1776



```python
ages = {'Sam':5, 'Frank':10, 'Spike':7}
```


```python
pd.Series(ages)
```




    Sam       5
    Frank    10
    Spike     7
    dtype: int64




```python
type(pd.Series(ages))
```




    pandas.core.series.Series




```python
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100, 'China': 500, 'India': 210, 'USA': 260}
```


```python
sales_q1 = pd.Series(q1)
sales_q2 = pd.Series(q2)
```


```python
print(sales_q1)
print(sales_q2)
```

    Japan     80
    China    450
    India    200
    USA      250
    dtype: int64
    Brazil    100
    China     500
    India     210
    USA       260
    dtype: int64



```python
print(sales_q1['Japan'])
print(sales_q1[0])
```

    80
    80



```python
sales_q1.keys()
```




    Index(['Japan', 'China', 'India', 'USA'], dtype='object')




```python
sales_q1 * 2
```




    Japan    160
    China    900
    India    400
    USA      500
    dtype: int64




```python
sales_q1 + sales_q2
```




    Brazil      NaN
    China     950.0
    India     410.0
    Japan       NaN
    USA       510.0
    dtype: float64




```python
sales_q1.add(sales_q2,fill_value=0)
```




    Brazil    100.0
    China     950.0
    India     410.0
    Japan      80.0
    USA       510.0
    dtype: float64




```python

```
