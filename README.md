# EXP 7 - ARMA IN PYTHON

## AIM:
To implement ARIMA model in python.

## ALGORITHM:

1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.


## PROGRAM:
python
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1,0.33])
ma1 = np.array([1,0.9])
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)

## OUTPUT:

### SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/611d4a31-4430-4140-a28d-285324b94a0b)

#### Partial Autocorrelation
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/f626ba2a-2ca8-4256-8113-b047b68c81e9)


#### Autocorrelation
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/83024171-aec3-4f93-8ae0-212fa051d1b0)

### SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/3b73df8e-b70f-4e86-b105-299ac438ca88)

#### Partial Autocorrelation
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/e9fb3ee5-4ee7-43fa-8246-4a842e4ec2c6)

#### Autocorrelation
![image](https://github.com/Aashima02/ARIMA-in-Python/assets/93427086/2e2f51e6-450a-456a-88ac-7bff4f1dd08c)


## RESULT:
Thus, a python program is created to implement ARMA.
