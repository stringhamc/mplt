# MPLT (My Plot)

A few wrappers around matplotlib easing some common plots that Signal processing
engineers like to use, such as plotting complex signals (real and imaginary or magnitude and phase).

## Usage
Note that mplt import all of pyplot's attributes, so there is no need to 
import matplotlib.pyplot as well

```python
import mplt
mplt.myfig('test')
mplt.implot(np.exp(1j*np.linspace(0,5,200)))
```
