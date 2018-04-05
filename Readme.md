
# Problem Description


A major problem faced by businesses in asset-heavy industries such as manufacturing is the significant costs that are associated with delays in the production process due to mechanical problems. Most of these businesses are interested in predicting these problems in advance so that they can proactively prevent the problems before they occur which will reduce the costly impact caused by downtime. Please refer to the playbook for predictive maintenance for a detailed explanation of common use cases in predictive maintenance and modelling approaches.

In this notebook, we follow the ideas from the playbook referenced above and aim to provide the steps of implementing a predictive model for a scenario which is based on a synthesis of multiple real-world business problems. This example brings together common data elements observed among many predictive maintenance use cases and the data itself is created by data simulation methods.

The business problem for this example is about predicting problems caused by component failures such that the question "What is the probability that a machine will fail in the near future due to a failure of a certain component?" can be answered. The problem is formatted as a multi-class classification problem and a machine learning algorithm is used to create the predictive model that learns from historical data collected from machines. In the following sections, we go through the steps of implementing such a model which are feature engineering, label construction, training and evaluation. First, we start by explaining the data sources in the next section.
# Data Sources

Common data sources for predictive maintenance problems are :

* **Failure history:** The failure history of a machine or component within the machine.
* **Maintenance history:** The repair history of a machine, e.g. error codes, previous maintenance activities or component replacements.
* **Machine conditions and usage:** The operating conditions of a machine e.g. data collected from sensors.
* **Machine features:** The features of a machine, e.g. engine size, make and model, location.
* **Operator features:** The features of the operator, e.g. gender, past experience
The data for this example comes from 4 different sources which are real-time telemetry data collected from machines, error messages, historical maintenance records that include failures and machine information such as type and age.


```python
import pandas as pd

telemetry = pd.read_csv('PdM_telemetry.csv')
errors = pd.read_csv('PdM_errors.csv')
maint = pd.read_csv('PdM_maint.csv')
failures = pd.read_csv('PdM_failures.csv')
machines = pd.read_csv('PdM_machines.csv')
```


```python
# format datetime field which comes in as string
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

print("Total number of telemetry records: %d" % len(telemetry.index))
print(telemetry.head())
telemetry.describe()
```

    Total number of telemetry records: 876100
                 datetime  machineID        volt      rotate    pressure  \
    0 2015-01-01 06:00:00          1  176.217853  418.504078  113.077935   
    1 2015-01-01 07:00:00          1  162.879223  402.747490   95.460525   
    2 2015-01-01 08:00:00          1  170.989902  527.349825   75.237905   
    3 2015-01-01 09:00:00          1  162.462833  346.149335  109.248561   
    4 2015-01-01 10:00:00          1  157.610021  435.376873  111.886648   
    
       vibration  
    0  45.087686  
    1  43.413973  
    2  34.178847  
    3  41.122144  
    4  25.990511  
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>volt</th>
      <th>rotate</th>
      <th>pressure</th>
      <th>vibration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.500000</td>
      <td>170.777736</td>
      <td>446.605119</td>
      <td>100.858668</td>
      <td>40.385007</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.866087</td>
      <td>15.509114</td>
      <td>52.673886</td>
      <td>11.048679</td>
      <td>5.370361</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>97.333604</td>
      <td>138.432075</td>
      <td>51.237106</td>
      <td>14.877054</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.750000</td>
      <td>160.304927</td>
      <td>412.305714</td>
      <td>93.498181</td>
      <td>36.777299</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.500000</td>
      <td>170.607338</td>
      <td>447.558150</td>
      <td>100.425559</td>
      <td>40.237247</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.250000</td>
      <td>181.004493</td>
      <td>482.176600</td>
      <td>107.555231</td>
      <td>43.784938</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>255.124717</td>
      <td>695.020984</td>
      <td>185.951998</td>
      <td>76.791072</td>
    </tr>
  </tbody>
</table>
</div>



#### **Telemetry**
The first data source is the telemetry time-series data which consists of **voltage, rotation, pressure, and vibration** measurements collected from 100 machines in **real time averaged over every hour collected during the year 2015**. Below, we display the first 10 records in the dataset. A summary of the whole dataset is also provided.


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

plot_df = telemetry.loc[(telemetry['machineID'] == 1) & 
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) & 
                        (telemetry['datetime'] <pd.to_datetime('2015-02-01')),
                        ['datetime','volt']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['datetime'], plot_df['volt'])
plt.ylabel('voltage')

# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Date')
```




    <matplotlib.text.Text at 0x2564dac0780>




![png](output_4_1.png)


#### **Errors**
The second major data source is the error logs. These are **non-breaking errors thrown while the machine is still operational and do not constitute as failures.** The **error date and times** are rounded to the closest hour since the telemetry data is collected at an hourly rate.


```python
# format of datetime field which comes in as string
errors['datetime'] = pd.to_datetime(errors['datetime'],format = '%Y-%m-%d %H:%M:%S')
errors['errorID'] = errors['errorID'].astype('category')
print("Total Number of error records: %d" %len(errors.index))
errors.head()
```

    Total Number of error records: 3919
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>errorID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-03 07:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-03 20:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-04 06:00:00</td>
      <td>1</td>
      <td>error5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-10 15:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-22 10:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
errors['errorID'].value_counts().plot(kind='bar')
plt.ylabel('Count')
errors['errorID'].value_counts()
```




    error1    1010
    error2     988
    error3     838
    error4     727
    error5     356
    Name: errorID, dtype: int64




![png](output_7_1.png)


#### **Maintenance**
These are the **scheduled and unscheduled** maintenance records which correspond to both **regular inspection of components as well as failures.** A **record is generated if a component is replaced during the scheduled inspection or replaced due to a breakdown.** The **records that are created due to breakdowns will be called failures** which is explained in the later sections. Maintenance data has both 2014 and 2015 records.


```python
maint['datetime'] = pd.to_datetime(maint['datetime'], format='%Y-%m-%d %H:%M:%S')
maint['comp'] = maint['comp'].astype('category')
print("Total Number of maintenance Records: %d" %len(maint.index))
maint.head()
```

    Total Number of maintenance Records: 3286
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>comp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-06-01 06:00:00</td>
      <td>1</td>
      <td>comp2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-07-16 06:00:00</td>
      <td>1</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-07-31 06:00:00</td>
      <td>1</td>
      <td>comp3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-13 06:00:00</td>
      <td>1</td>
      <td>comp1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-05 06:00:00</td>
      <td>1</td>
      <td>comp4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("darkgrid")
plt.figure(figsize=(10, 4))
maint['comp'].value_counts().plot(kind='bar')
plt.ylabel('Count')
maint['comp'].value_counts()
```




    comp2    863
    comp4    811
    comp3    808
    comp1    804
    Name: comp, dtype: int64




![png](output_10_1.png)


#### **Machines**
This data set includes some information about the machines: model type and age (years in service).


```python
machines['model'] = machines['model'].astype('category')

print("Total number of machines: %d" % len(machines.index))
machines.head()
```

    Total number of machines: 100
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>model</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>model3</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>model4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>model3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>model3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>model3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("darkgrid")
plt.figure(figsize=(15, 6))
_, bins, _ = plt.hist([machines.loc[machines['model'] == 'model1', 'age'],
                       machines.loc[machines['model'] == 'model2', 'age'],
                       machines.loc[machines['model'] == 'model3', 'age'],
                       machines.loc[machines['model'] == 'model4', 'age']],
                       20, stacked=True, label=['model1', 'model2', 'model3', 'model4'])
plt.xlabel('Age (yrs)')
plt.ylabel('Count')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2564dda9898>




![png](output_13_1.png)


#### **Failures**
These are the records of component replacements **due to failures.** Each record has a **date and time, machine ID, and failed component type.**


```python
# format datetime field which comes in as string
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
failures['failure'] = failures['failure'].astype('category')

print("Total number of failures: %d" % len(failures.index))
failures.head()
```

    Total number of failures: 761
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>failure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-05 06:00:00</td>
      <td>1</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-03-06 06:00:00</td>
      <td>1</td>
      <td>comp1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-04-20 06:00:00</td>
      <td>1</td>
      <td>comp2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-06-19 06:00:00</td>
      <td>1</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-09-02 06:00:00</td>
      <td>1</td>
      <td>comp4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("darkgrid")
plt.figure(figsize=(15, 4))
failures['failure'].value_counts().plot(kind='bar')
plt.ylabel('Count')
failures['failure'].value_counts()
```




    comp2    259
    comp1    192
    comp4    179
    comp3    131
    Name: failure, dtype: int64




![png](output_16_1.png)


## Feature Engineering
The first step in predictive maintenance applications is feature engineering which requires bringing the different data sources together to create features that best describe a machines's health condition at a given point in time. In the next sections, several feature engineering methods are used to create features based on the properties of each data source.

### Lag Features from Telemetry
Telemetry data almost always comes with time-stamps which makes it suitable for calculating lagging features. A common method is to pick a window size for the lag features to be created and compute rolling aggregate measures such as mean, standard deviation, minimum, maximum, etc. to represent the short term history of the telemetry over the lag window. In the following, rolling mean and standard deviation of the telemetry data over the last 3 hour lag window is calculated for every 3 hours.


```python
# Calculate mean values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='mean').unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)
```

    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:8: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).mean()
      
    


```python
# Calculate mean values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='mean').unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)

# repeat for standard deviation
temp = []
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='std').unstack())
telemetry_sd_3h = pd.concat(temp, axis=1)
telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
telemetry_sd_3h.reset_index(inplace=True)

telemetry_mean_3h.head()
```

    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:8: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).mean()
      
    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:19: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).std()
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-01-01 09:00:00</td>
      <td>170.028993</td>
      <td>449.533798</td>
      <td>94.592122</td>
      <td>40.893502</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-01-01 12:00:00</td>
      <td>164.192565</td>
      <td>403.949857</td>
      <td>105.687417</td>
      <td>34.255891</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-01-01 15:00:00</td>
      <td>168.134445</td>
      <td>435.781707</td>
      <td>107.793709</td>
      <td>41.239405</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-01-01 18:00:00</td>
      <td>165.514453</td>
      <td>430.472823</td>
      <td>101.703289</td>
      <td>40.373739</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-01-01 21:00:00</td>
      <td>168.809347</td>
      <td>437.111120</td>
      <td>90.911060</td>
      <td>41.738542</td>
    </tr>
  </tbody>
</table>
</div>




For capturing a longer term effect, 24 hour lag features are also calculated as below.


```python
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.rolling_mean(pd.pivot_table(telemetry,
                                               index='datetime',
                                               columns='machineID',
                                               values=col), window=24).resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
telemetry_mean_24h = pd.concat(temp, axis=1)
telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
telemetry_mean_24h.reset_index(inplace=True)
telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

# repeat for standard deviation
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.rolling_std(pd.pivot_table(telemetry,
                                               index='datetime',
                                               columns='machineID',
                                               values=col), window=24).resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
telemetry_sd_24h = pd.concat(temp, axis=1)
telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
telemetry_sd_24h.reset_index(inplace=True)

# Notice that a 24h rolling average is not available at the earliest timepoints
telemetry_mean_24h.head(10)
```

    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:7: FutureWarning: pd.rolling_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.rolling(window=24,center=False).mean()
      import sys
    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:10: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).first()
      # Remove the CWD from sys.path while we load stuff.
    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:23: FutureWarning: pd.rolling_std is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.rolling(window=24,center=False).std()
    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:26: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).first()
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_24h</th>
      <th>rotatemean_24h</th>
      <th>pressuremean_24h</th>
      <th>vibrationmean_24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2015-01-02 06:00:00</td>
      <td>169.733809</td>
      <td>445.179865</td>
      <td>96.797113</td>
      <td>40.385160</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2015-01-02 09:00:00</td>
      <td>170.614862</td>
      <td>446.364859</td>
      <td>96.849785</td>
      <td>39.736826</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2015-01-02 12:00:00</td>
      <td>169.893965</td>
      <td>447.009407</td>
      <td>97.715600</td>
      <td>39.498374</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2015-01-02 15:00:00</td>
      <td>171.243444</td>
      <td>444.233563</td>
      <td>96.666060</td>
      <td>40.229370</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2015-01-02 18:00:00</td>
      <td>170.792486</td>
      <td>448.440437</td>
      <td>95.766838</td>
      <td>40.055214</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>2015-01-02 21:00:00</td>
      <td>170.556674</td>
      <td>452.267095</td>
      <td>98.065860</td>
      <td>40.033247</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2015-01-03 00:00:00</td>
      <td>168.460525</td>
      <td>451.031783</td>
      <td>99.273286</td>
      <td>38.903462</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2015-01-03 03:00:00</td>
      <td>169.772951</td>
      <td>447.502464</td>
      <td>99.005946</td>
      <td>39.389725</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2015-01-03 06:00:00</td>
      <td>170.900562</td>
      <td>453.864597</td>
      <td>100.877342</td>
      <td>38.696225</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2015-01-03 09:00:00</td>
      <td>169.533156</td>
      <td>454.785072</td>
      <td>100.050567</td>
      <td>39.449734</td>
    </tr>
  </tbody>
</table>
</div>



Next, the columns of the feature datasets created earlier are merged to create the final feature set from telemetry.


```python
# merge columns of feature sets created earlier
telemetry_feat = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.ix[:, 2:6],
                            telemetry_mean_24h.ix[:, 2:6],
                            telemetry_sd_24h.ix[:, 2:6]], axis=1).dropna()
telemetry_feat.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>voltmean_24h</th>
      <th>rotatemean_24h</th>
      <th>pressuremean_24h</th>
      <th>vibrationmean_24h</th>
      <th>voltsd_24h</th>
      <th>rotatesd_24h</th>
      <th>pressuresd_24h</th>
      <th>vibrationsd_24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.380935</td>
      <td>170.774427</td>
      <td>446.609386</td>
      <td>100.858340</td>
      <td>40.383609</td>
      <td>13.300173</td>
      <td>44.453951</td>
      <td>8.885780</td>
      <td>4.440575</td>
      <td>170.775661</td>
      <td>446.609874</td>
      <td>100.857574</td>
      <td>40.383881</td>
      <td>14.919452</td>
      <td>49.950788</td>
      <td>10.046380</td>
      <td>5.002089</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.798424</td>
      <td>9.498824</td>
      <td>33.119738</td>
      <td>7.411701</td>
      <td>3.475512</td>
      <td>6.966389</td>
      <td>23.214291</td>
      <td>4.656364</td>
      <td>2.319989</td>
      <td>4.720237</td>
      <td>18.070458</td>
      <td>4.737293</td>
      <td>2.058059</td>
      <td>2.261097</td>
      <td>7.684305</td>
      <td>1.713206</td>
      <td>0.799599</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>125.532506</td>
      <td>211.811184</td>
      <td>72.118639</td>
      <td>26.569635</td>
      <td>0.025509</td>
      <td>0.078991</td>
      <td>0.027417</td>
      <td>0.015278</td>
      <td>155.812721</td>
      <td>266.010419</td>
      <td>91.057429</td>
      <td>35.060087</td>
      <td>6.380619</td>
      <td>18.385248</td>
      <td>4.145308</td>
      <td>2.144863</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>164.447794</td>
      <td>427.564793</td>
      <td>96.239534</td>
      <td>38.147458</td>
      <td>8.028675</td>
      <td>26.906319</td>
      <td>5.369959</td>
      <td>2.684556</td>
      <td>168.072275</td>
      <td>441.542561</td>
      <td>98.669734</td>
      <td>39.354077</td>
      <td>13.359069</td>
      <td>44.669022</td>
      <td>8.924165</td>
      <td>4.460675</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>170.432407</td>
      <td>448.380260</td>
      <td>100.235357</td>
      <td>40.145874</td>
      <td>12.495542</td>
      <td>41.793798</td>
      <td>8.345801</td>
      <td>4.173704</td>
      <td>170.212704</td>
      <td>449.206885</td>
      <td>100.099533</td>
      <td>40.072618</td>
      <td>14.854186</td>
      <td>49.617459</td>
      <td>9.921332</td>
      <td>4.958793</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.000000</td>
      <td>176.610017</td>
      <td>468.443933</td>
      <td>104.406534</td>
      <td>42.226898</td>
      <td>17.688520</td>
      <td>59.092354</td>
      <td>11.789358</td>
      <td>5.898512</td>
      <td>172.462228</td>
      <td>456.366349</td>
      <td>101.613047</td>
      <td>40.833112</td>
      <td>16.395372</td>
      <td>54.826993</td>
      <td>10.980250</td>
      <td>5.484430</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>241.420717</td>
      <td>586.682904</td>
      <td>162.309656</td>
      <td>69.311324</td>
      <td>58.444332</td>
      <td>179.903039</td>
      <td>35.659369</td>
      <td>18.305595</td>
      <td>220.782618</td>
      <td>499.096975</td>
      <td>152.310351</td>
      <td>61.932124</td>
      <td>27.664538</td>
      <td>103.819404</td>
      <td>28.654103</td>
      <td>12.325783</td>
    </tr>
  </tbody>
</table>
</div>




```python
telemetry_feat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>voltmean_24h</th>
      <th>rotatemean_24h</th>
      <th>pressuremean_24h</th>
      <th>vibrationmean_24h</th>
      <th>voltsd_24h</th>
      <th>rotatesd_24h</th>
      <th>pressuresd_24h</th>
      <th>vibrationsd_24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2015-01-02 06:00:00</td>
      <td>180.133784</td>
      <td>440.608320</td>
      <td>94.137969</td>
      <td>41.551544</td>
      <td>21.322735</td>
      <td>48.770512</td>
      <td>2.135684</td>
      <td>10.037208</td>
      <td>169.733809</td>
      <td>445.179865</td>
      <td>96.797113</td>
      <td>40.385160</td>
      <td>15.726970</td>
      <td>39.648116</td>
      <td>11.904700</td>
      <td>5.601191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2015-01-02 09:00:00</td>
      <td>176.364293</td>
      <td>439.349655</td>
      <td>101.553209</td>
      <td>36.105580</td>
      <td>18.952210</td>
      <td>51.329636</td>
      <td>13.789279</td>
      <td>6.737739</td>
      <td>170.614862</td>
      <td>446.364859</td>
      <td>96.849785</td>
      <td>39.736826</td>
      <td>15.635083</td>
      <td>41.828592</td>
      <td>11.326412</td>
      <td>5.583521</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2015-01-02 12:00:00</td>
      <td>160.384568</td>
      <td>424.385316</td>
      <td>99.598722</td>
      <td>36.094637</td>
      <td>13.047080</td>
      <td>13.702496</td>
      <td>9.988609</td>
      <td>1.639962</td>
      <td>169.893965</td>
      <td>447.009407</td>
      <td>97.715600</td>
      <td>39.498374</td>
      <td>13.995465</td>
      <td>40.843882</td>
      <td>11.036546</td>
      <td>5.561553</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2015-01-02 15:00:00</td>
      <td>170.472461</td>
      <td>442.933997</td>
      <td>102.380586</td>
      <td>40.483002</td>
      <td>16.642354</td>
      <td>56.290447</td>
      <td>3.305739</td>
      <td>8.854145</td>
      <td>171.243444</td>
      <td>444.233563</td>
      <td>96.666060</td>
      <td>40.229370</td>
      <td>13.100364</td>
      <td>43.409841</td>
      <td>10.972862</td>
      <td>6.068674</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2015-01-02 18:00:00</td>
      <td>163.263806</td>
      <td>468.937558</td>
      <td>102.726648</td>
      <td>40.921802</td>
      <td>17.424688</td>
      <td>38.680380</td>
      <td>9.105775</td>
      <td>3.060781</td>
      <td>170.792486</td>
      <td>448.440437</td>
      <td>95.766838</td>
      <td>40.055214</td>
      <td>13.808489</td>
      <td>43.742304</td>
      <td>10.988704</td>
      <td>7.286129</td>
    </tr>
  </tbody>
</table>
</div>



### Lag Features from Errors
Like telemetry data, errors come with timestamps. An important difference is that the **error IDs are categorical values** and **should not be averaged over time intervals like the telemetry measurements.** Instead, we count the number of errors of each type in a **lagging window. We begin by reformatting the error data** to have one entry per machine per time at which at least one error occurred:


```python
errors
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>errorID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-03 07:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-03 20:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-04 06:00:00</td>
      <td>1</td>
      <td>error5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-10 15:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-22 10:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-01-25 15:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-01-27 04:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-03-03 22:00:00</td>
      <td>1</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-03-05 06:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-03-20 18:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-03-26 01:00:00</td>
      <td>1</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-03-31 23:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-04-19 06:00:00</td>
      <td>1</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2015-04-19 06:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-04-29 19:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015-05-04 23:00:00</td>
      <td>1</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2015-05-12 09:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015-05-21 07:00:00</td>
      <td>1</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-05-24 02:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2015-05-25 05:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2015-06-09 06:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2015-06-18 06:00:00</td>
      <td>1</td>
      <td>error5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2015-06-23 10:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2015-08-23 19:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-08-30 01:00:00</td>
      <td>1</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2015-09-01 06:00:00</td>
      <td>1</td>
      <td>error5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2015-09-13 17:00:00</td>
      <td>1</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015-09-15 06:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2015-10-01 23:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015-10-15 05:00:00</td>
      <td>1</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3889</th>
      <td>2015-01-16 00:00:00</td>
      <td>100</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>3890</th>
      <td>2015-02-01 10:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3891</th>
      <td>2015-02-11 06:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3892</th>
      <td>2015-02-12 21:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3893</th>
      <td>2015-03-08 15:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3894</th>
      <td>2015-04-27 04:00:00</td>
      <td>100</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>3895</th>
      <td>2015-04-27 22:00:00</td>
      <td>100</td>
      <td>error5</td>
    </tr>
    <tr>
      <th>3896</th>
      <td>2015-05-16 23:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3897</th>
      <td>2015-05-17 13:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3898</th>
      <td>2015-05-22 02:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3899</th>
      <td>2015-07-05 16:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3900</th>
      <td>2015-07-19 01:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3901</th>
      <td>2015-08-14 16:00:00</td>
      <td>100</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>3902</th>
      <td>2015-08-30 15:00:00</td>
      <td>100</td>
      <td>error4</td>
    </tr>
    <tr>
      <th>3903</th>
      <td>2015-09-09 06:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3904</th>
      <td>2015-09-14 23:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3905</th>
      <td>2015-10-03 05:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3906</th>
      <td>2015-10-09 07:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3907</th>
      <td>2015-10-17 02:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3908</th>
      <td>2015-10-17 12:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>2015-10-24 23:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>2015-10-27 21:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3911</th>
      <td>2015-11-05 02:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3912</th>
      <td>2015-11-07 17:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3913</th>
      <td>2015-11-12 01:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3914</th>
      <td>2015-11-21 08:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3915</th>
      <td>2015-12-04 02:00:00</td>
      <td>100</td>
      <td>error1</td>
    </tr>
    <tr>
      <th>3916</th>
      <td>2015-12-08 06:00:00</td>
      <td>100</td>
      <td>error2</td>
    </tr>
    <tr>
      <th>3917</th>
      <td>2015-12-08 06:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
    <tr>
      <th>3918</th>
      <td>2015-12-22 03:00:00</td>
      <td>100</td>
      <td>error3</td>
    </tr>
  </tbody>
</table>
<p>3919 rows × 3 columns</p>
</div>




```python
# create a column for each error type
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
error_count
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
error_count.head(13)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>error1</th>
      <th>error2</th>
      <th>error3</th>
      <th>error4</th>
      <th>error5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-03 07:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-03 20:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-04 06:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-10 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-22 10:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-01-25 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-01-27 04:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-03-03 22:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-03-05 06:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-03-20 18:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-03-26 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-03-31 23:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-04-19 06:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# combine errors for a given machine in a given hour
error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
error_count.head(13)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>error1</th>
      <th>error2</th>
      <th>error3</th>
      <th>error4</th>
      <th>error5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-01-03 07:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-01-03 20:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-01-04 06:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-01-10 15:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-01-22 10:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2015-01-25 15:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2015-01-27 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2015-03-03 22:00:00</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2015-03-05 06:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2015-03-20 18:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2015-03-26 01:00:00</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2015-03-31 23:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>2015-04-19 06:00:00</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
error_count.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>error1</th>
      <th>error2</th>
      <th>error3</th>
      <th>error4</th>
      <th>error5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.500000</td>
      <td>0.001153</td>
      <td>0.001128</td>
      <td>0.000957</td>
      <td>0.000830</td>
      <td>0.000406</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.866087</td>
      <td>0.033934</td>
      <td>0.033563</td>
      <td>0.030913</td>
      <td>0.028795</td>
      <td>0.020154</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can compute the total number of errors of each type over the last 24 hours, for timepoints taken every three hours:


```python
temp = []
fields = ['error%d' % i for i in range(1,6)]
for col in fields:
    temp.append(pd.rolling_sum(pd.pivot_table(error_count,
                                               index='datetime',
                                               columns='machineID',
                                               values=col), window=24).resample('3H',
                                                                             closed='left',
                                                                             label='right',
                                                                             how='first').unstack())
error_count = pd.concat(temp, axis=1)
error_count.columns = [i + 'count' for i in fields]
error_count.reset_index(inplace=True)
error_count = error_count.dropna()
error_count.describe()
```

    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:7: FutureWarning: pd.rolling_sum is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.rolling(window=24,center=False).sum()
      import sys
    C:\Program Files\Microsoft\ML Server\PYTHON_SERVER\lib\site-packages\ipykernel_launcher.py:10: FutureWarning: how in .resample() is deprecated
    the new syntax is .resample(...).first()
      # Remove the CWD from sys.path while we load stuff.
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>error1count</th>
      <th>error2count</th>
      <th>error3count</th>
      <th>error4count</th>
      <th>error5count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>291400.00000</td>
      <td>291400.000000</td>
      <td>291400.000000</td>
      <td>291400.000000</td>
      <td>291400.000000</td>
      <td>291400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.50000</td>
      <td>0.027649</td>
      <td>0.027069</td>
      <td>0.022907</td>
      <td>0.019904</td>
      <td>0.009753</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.86612</td>
      <td>0.166273</td>
      <td>0.164429</td>
      <td>0.151453</td>
      <td>0.140820</td>
      <td>0.098797</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.75000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.50000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.25000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.00000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
error_count.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>error1count</th>
      <th>error2count</th>
      <th>error3count</th>
      <th>error4count</th>
      <th>error5count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2015-01-02 06:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2015-01-02 09:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2015-01-02 12:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2015-01-02 15:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2015-01-02 18:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Days Since Last Replacement from Maintenance
A crucial data set in this example is the maintenance records which contain the information of component replacement records. Possible features from this data set can be, for example, the number of replacements of each component in the last 3 months to incorporate the frequency of replacements. However, more relevent information would be to calculate how long it has been since a component is last replaced as that would be expected to correlate better with component failures since the longer a component is used, the more degradation should be expected.

As a side note, creating lagging features from maintenance data is not as straightforward as for telemetry and errors, so the features from this data are generated in a more custom way. This type of ad-hoc feature engineering is very common in predictive maintenance since domain knowledge plays a big role in understanding the predictors of a problem. In the following, the days since last component replacement are calculated for each component type as features from the maintenance data.


```python
import numpy as np

# create a column for each error type
comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

# combine repairs for a given machine in a given hour
comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

# add timepoints where no components were replaced
comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                      on=['datetime', 'machineID'],
                                                      how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
    # convert indicator to most recent date of component change
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
    
    # forward-fill the most-recent date of component change
    comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

# remove dates in 2014 (may have NaN or future component change dates)    
comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

# replace dates of most recent component change with days since most recent component change
for comp in components:
    comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')
    
comp_rep.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
      <td>876100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.500000</td>
      <td>53.525185</td>
      <td>51.540806</td>
      <td>52.725962</td>
      <td>53.834191</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.866087</td>
      <td>62.491679</td>
      <td>59.269254</td>
      <td>58.873114</td>
      <td>59.707978</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.750000</td>
      <td>13.291667</td>
      <td>12.125000</td>
      <td>13.125000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.500000</td>
      <td>32.791667</td>
      <td>29.666667</td>
      <td>32.291667</td>
      <td>32.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.250000</td>
      <td>68.708333</td>
      <td>66.541667</td>
      <td>67.333333</td>
      <td>70.458333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>491.958333</td>
      <td>348.958333</td>
      <td>370.958333</td>
      <td>394.958333</td>
    </tr>
  </tbody>
</table>
</div>




```python
comp_rep.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01 06:00:00</td>
      <td>1</td>
      <td>19.000000</td>
      <td>214.000000</td>
      <td>154.000000</td>
      <td>169.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01 07:00:00</td>
      <td>1</td>
      <td>19.041667</td>
      <td>214.041667</td>
      <td>154.041667</td>
      <td>169.041667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01 08:00:00</td>
      <td>1</td>
      <td>19.083333</td>
      <td>214.083333</td>
      <td>154.083333</td>
      <td>169.083333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01 09:00:00</td>
      <td>1</td>
      <td>19.125000</td>
      <td>214.125000</td>
      <td>154.125000</td>
      <td>169.125000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01 10:00:00</td>
      <td>1</td>
      <td>19.166667</td>
      <td>214.166667</td>
      <td>154.166667</td>
      <td>169.166667</td>
    </tr>
  </tbody>
</table>
</div>



## Machine Features
The machine features can be used without further modification. These include descriptive information about the type of each machine and its age (number of years in service). If the age information had been recorded as a "first use date" for each machine, a transformation would have been necessary to turn those into a numeric values indicating the years in service.

Lastly, we merge all the feature data sets we created earlier to get the final feature matrix.


```python

```


```python
telemetry_feat
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>voltmean_24h</th>
      <th>rotatemean_24h</th>
      <th>pressuremean_24h</th>
      <th>vibrationmean_24h</th>
      <th>voltsd_24h</th>
      <th>rotatesd_24h</th>
      <th>pressuresd_24h</th>
      <th>vibrationsd_24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2015-01-02 06:00:00</td>
      <td>180.133784</td>
      <td>440.608320</td>
      <td>94.137969</td>
      <td>41.551544</td>
      <td>21.322735</td>
      <td>48.770512</td>
      <td>2.135684</td>
      <td>10.037208</td>
      <td>169.733809</td>
      <td>445.179865</td>
      <td>96.797113</td>
      <td>40.385160</td>
      <td>15.726970</td>
      <td>39.648116</td>
      <td>11.904700</td>
      <td>5.601191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2015-01-02 09:00:00</td>
      <td>176.364293</td>
      <td>439.349655</td>
      <td>101.553209</td>
      <td>36.105580</td>
      <td>18.952210</td>
      <td>51.329636</td>
      <td>13.789279</td>
      <td>6.737739</td>
      <td>170.614862</td>
      <td>446.364859</td>
      <td>96.849785</td>
      <td>39.736826</td>
      <td>15.635083</td>
      <td>41.828592</td>
      <td>11.326412</td>
      <td>5.583521</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2015-01-02 12:00:00</td>
      <td>160.384568</td>
      <td>424.385316</td>
      <td>99.598722</td>
      <td>36.094637</td>
      <td>13.047080</td>
      <td>13.702496</td>
      <td>9.988609</td>
      <td>1.639962</td>
      <td>169.893965</td>
      <td>447.009407</td>
      <td>97.715600</td>
      <td>39.498374</td>
      <td>13.995465</td>
      <td>40.843882</td>
      <td>11.036546</td>
      <td>5.561553</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2015-01-02 15:00:00</td>
      <td>170.472461</td>
      <td>442.933997</td>
      <td>102.380586</td>
      <td>40.483002</td>
      <td>16.642354</td>
      <td>56.290447</td>
      <td>3.305739</td>
      <td>8.854145</td>
      <td>171.243444</td>
      <td>444.233563</td>
      <td>96.666060</td>
      <td>40.229370</td>
      <td>13.100364</td>
      <td>43.409841</td>
      <td>10.972862</td>
      <td>6.068674</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2015-01-02 18:00:00</td>
      <td>163.263806</td>
      <td>468.937558</td>
      <td>102.726648</td>
      <td>40.921802</td>
      <td>17.424688</td>
      <td>38.680380</td>
      <td>9.105775</td>
      <td>3.060781</td>
      <td>170.792486</td>
      <td>448.440437</td>
      <td>95.766838</td>
      <td>40.055214</td>
      <td>13.808489</td>
      <td>43.742304</td>
      <td>10.988704</td>
      <td>7.286129</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>2015-01-02 21:00:00</td>
      <td>163.278466</td>
      <td>446.493166</td>
      <td>104.387585</td>
      <td>38.068116</td>
      <td>21.580492</td>
      <td>41.380958</td>
      <td>20.725597</td>
      <td>6.932127</td>
      <td>170.556674</td>
      <td>452.267095</td>
      <td>98.065860</td>
      <td>40.033247</td>
      <td>14.187985</td>
      <td>40.676672</td>
      <td>11.942227</td>
      <td>8.723238</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2015-01-03 00:00:00</td>
      <td>172.191198</td>
      <td>434.214692</td>
      <td>93.747282</td>
      <td>39.716482</td>
      <td>16.369836</td>
      <td>14.636041</td>
      <td>18.817326</td>
      <td>3.426997</td>
      <td>168.460525</td>
      <td>451.031783</td>
      <td>99.273286</td>
      <td>38.903462</td>
      <td>13.707794</td>
      <td>40.509184</td>
      <td>10.141026</td>
      <td>8.634082</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2015-01-03 03:00:00</td>
      <td>175.210027</td>
      <td>504.845430</td>
      <td>108.512153</td>
      <td>37.763933</td>
      <td>5.991921</td>
      <td>16.062702</td>
      <td>6.382608</td>
      <td>3.449468</td>
      <td>169.772951</td>
      <td>447.502464</td>
      <td>99.005946</td>
      <td>39.389725</td>
      <td>11.818603</td>
      <td>44.468516</td>
      <td>9.444955</td>
      <td>8.332673</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2015-01-03 06:00:00</td>
      <td>181.690108</td>
      <td>472.783187</td>
      <td>93.395164</td>
      <td>38.621099</td>
      <td>11.514450</td>
      <td>47.880443</td>
      <td>2.177029</td>
      <td>7.670520</td>
      <td>170.900562</td>
      <td>453.864597</td>
      <td>100.877342</td>
      <td>38.696225</td>
      <td>12.069391</td>
      <td>46.669661</td>
      <td>8.609526</td>
      <td>8.089348</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2015-01-03 09:00:00</td>
      <td>172.382935</td>
      <td>505.141261</td>
      <td>98.524373</td>
      <td>49.965572</td>
      <td>7.065150</td>
      <td>56.849540</td>
      <td>5.230039</td>
      <td>2.687565</td>
      <td>169.533156</td>
      <td>454.785072</td>
      <td>100.050567</td>
      <td>39.449734</td>
      <td>12.755234</td>
      <td>44.016114</td>
      <td>9.893704</td>
      <td>7.013132</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2015-01-03 12:00:00</td>
      <td>174.303858</td>
      <td>436.182686</td>
      <td>94.092681</td>
      <td>50.999589</td>
      <td>19.017196</td>
      <td>26.420163</td>
      <td>7.661944</td>
      <td>3.516734</td>
      <td>170.866013</td>
      <td>463.871291</td>
      <td>99.360632</td>
      <td>40.766639</td>
      <td>12.848646</td>
      <td>45.090576</td>
      <td>9.846662</td>
      <td>5.888262</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2015-01-03 15:00:00</td>
      <td>176.246348</td>
      <td>451.646684</td>
      <td>98.102389</td>
      <td>59.198241</td>
      <td>12.572504</td>
      <td>31.574383</td>
      <td>15.559351</td>
      <td>6.562087</td>
      <td>171.041651</td>
      <td>463.701291</td>
      <td>98.965877</td>
      <td>42.396850</td>
      <td>14.968351</td>
      <td>37.088898</td>
      <td>10.133452</td>
      <td>5.702356</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>2015-01-03 18:00:00</td>
      <td>158.433533</td>
      <td>453.900213</td>
      <td>98.878129</td>
      <td>46.851925</td>
      <td>5.136952</td>
      <td>21.216569</td>
      <td>11.400650</td>
      <td>2.688559</td>
      <td>171.244533</td>
      <td>464.320613</td>
      <td>98.853189</td>
      <td>44.608814</td>
      <td>17.058217</td>
      <td>36.617908</td>
      <td>9.867174</td>
      <td>5.743753</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>2015-01-03 21:00:00</td>
      <td>162.387954</td>
      <td>454.140377</td>
      <td>92.651129</td>
      <td>54.261635</td>
      <td>4.563331</td>
      <td>57.747656</td>
      <td>4.754203</td>
      <td>5.118076</td>
      <td>171.385039</td>
      <td>459.937314</td>
      <td>97.292157</td>
      <td>45.284751</td>
      <td>18.405763</td>
      <td>35.819938</td>
      <td>9.743769</td>
      <td>5.246435</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>2015-01-04 00:00:00</td>
      <td>174.243192</td>
      <td>394.998095</td>
      <td>99.829845</td>
      <td>46.930738</td>
      <td>6.268730</td>
      <td>29.167663</td>
      <td>10.564287</td>
      <td>6.822855</td>
      <td>171.880633</td>
      <td>461.437128</td>
      <td>96.786742</td>
      <td>47.311018</td>
      <td>18.249831</td>
      <td>42.055638</td>
      <td>10.961128</td>
      <td>5.093464</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>2015-01-04 03:00:00</td>
      <td>176.443361</td>
      <td>459.528820</td>
      <td>111.855296</td>
      <td>55.296056</td>
      <td>16.330285</td>
      <td>20.602657</td>
      <td>7.064583</td>
      <td>4.651468</td>
      <td>172.513202</td>
      <td>456.429165</td>
      <td>97.742700</td>
      <td>48.416442</td>
      <td>19.141287</td>
      <td>37.018824</td>
      <td>10.642956</td>
      <td>4.618287</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>2015-01-04 06:00:00</td>
      <td>186.092896</td>
      <td>451.641253</td>
      <td>107.989359</td>
      <td>55.308074</td>
      <td>13.489090</td>
      <td>62.185045</td>
      <td>5.118176</td>
      <td>4.904365</td>
      <td>172.686245</td>
      <td>453.387589</td>
      <td>99.304019</td>
      <td>51.158654</td>
      <td>18.887033</td>
      <td>36.997459</td>
      <td>11.042775</td>
      <td>5.195423</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>2015-01-04 09:00:00</td>
      <td>166.281848</td>
      <td>453.787824</td>
      <td>106.187582</td>
      <td>51.990080</td>
      <td>24.276228</td>
      <td>23.621315</td>
      <td>11.176731</td>
      <td>3.394073</td>
      <td>172.042428</td>
      <td>450.418764</td>
      <td>100.284484</td>
      <td>52.153213</td>
      <td>20.837993</td>
      <td>34.051825</td>
      <td>9.654971</td>
      <td>5.066388</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>2015-01-04 12:00:00</td>
      <td>175.412103</td>
      <td>445.450581</td>
      <td>100.887363</td>
      <td>54.251534</td>
      <td>34.918687</td>
      <td>11.001625</td>
      <td>10.580336</td>
      <td>2.921501</td>
      <td>171.219623</td>
      <td>443.802134</td>
      <td>102.358897</td>
      <td>52.854420</td>
      <td>21.298322</td>
      <td>36.054002</td>
      <td>9.885781</td>
      <td>5.246894</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>2015-01-04 15:00:00</td>
      <td>157.347716</td>
      <td>451.882075</td>
      <td>101.289380</td>
      <td>48.602686</td>
      <td>24.617739</td>
      <td>28.950883</td>
      <td>9.966729</td>
      <td>2.356486</td>
      <td>172.013443</td>
      <td>444.882018</td>
      <td>102.578580</td>
      <td>52.789794</td>
      <td>21.200183</td>
      <td>38.544116</td>
      <td>10.429692</td>
      <td>7.192434</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>2015-01-04 18:00:00</td>
      <td>176.450550</td>
      <td>446.033068</td>
      <td>84.521555</td>
      <td>47.638836</td>
      <td>8.071400</td>
      <td>76.511343</td>
      <td>2.636879</td>
      <td>4.108621</td>
      <td>170.176321</td>
      <td>445.069594</td>
      <td>102.359939</td>
      <td>51.518719</td>
      <td>18.814679</td>
      <td>40.547527</td>
      <td>11.133170</td>
      <td>7.556313</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>2015-01-04 21:00:00</td>
      <td>190.325814</td>
      <td>422.692565</td>
      <td>107.393234</td>
      <td>49.552856</td>
      <td>8.390777</td>
      <td>7.176553</td>
      <td>4.262645</td>
      <td>7.598552</td>
      <td>172.932248</td>
      <td>444.618018</td>
      <td>101.425508</td>
      <td>52.135905</td>
      <td>16.762469</td>
      <td>49.373445</td>
      <td>10.443534</td>
      <td>8.545739</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>2015-01-05 00:00:00</td>
      <td>169.985134</td>
      <td>458.929418</td>
      <td>91.494362</td>
      <td>54.882021</td>
      <td>9.451483</td>
      <td>12.052752</td>
      <td>3.685906</td>
      <td>6.621183</td>
      <td>175.121131</td>
      <td>443.916392</td>
      <td>102.130179</td>
      <td>51.653294</td>
      <td>17.435946</td>
      <td>43.819375</td>
      <td>10.830449</td>
      <td>8.809530</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>2015-01-05 03:00:00</td>
      <td>149.082619</td>
      <td>412.180336</td>
      <td>93.509785</td>
      <td>54.386079</td>
      <td>19.075952</td>
      <td>30.715081</td>
      <td>3.090266</td>
      <td>6.530610</td>
      <td>173.407255</td>
      <td>446.265950</td>
      <td>100.874614</td>
      <td>52.529450</td>
      <td>16.661364</td>
      <td>47.266846</td>
      <td>11.225440</td>
      <td>9.068824</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>2015-01-05 06:00:00</td>
      <td>185.782709</td>
      <td>439.531288</td>
      <td>99.413660</td>
      <td>51.558082</td>
      <td>14.495664</td>
      <td>45.663743</td>
      <td>4.289212</td>
      <td>7.330397</td>
      <td>170.757841</td>
      <td>440.958228</td>
      <td>98.716746</td>
      <td>51.746749</td>
      <td>17.863934</td>
      <td>44.895080</td>
      <td>10.675981</td>
      <td>7.475304</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>2015-01-05 09:00:00</td>
      <td>169.084809</td>
      <td>463.433785</td>
      <td>107.678774</td>
      <td>41.710336</td>
      <td>12.245544</td>
      <td>61.759107</td>
      <td>4.400233</td>
      <td>9.750017</td>
      <td>171.929104</td>
      <td>443.448775</td>
      <td>98.675590</td>
      <td>51.780445</td>
      <td>15.139300</td>
      <td>45.766081</td>
      <td>10.959268</td>
      <td>6.855778</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>2015-01-05 12:00:00</td>
      <td>165.518790</td>
      <td>449.743255</td>
      <td>110.377851</td>
      <td>38.952082</td>
      <td>23.170638</td>
      <td>45.762142</td>
      <td>14.009473</td>
      <td>0.797364</td>
      <td>170.908522</td>
      <td>443.069042</td>
      <td>98.830333</td>
      <td>49.679550</td>
      <td>13.985517</td>
      <td>42.542001</td>
      <td>11.050133</td>
      <td>4.842842</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>2015-01-05 15:00:00</td>
      <td>175.989642</td>
      <td>419.863490</td>
      <td>112.571146</td>
      <td>41.514254</td>
      <td>4.028327</td>
      <td>20.148499</td>
      <td>5.862629</td>
      <td>9.702498</td>
      <td>170.416326</td>
      <td>443.555122</td>
      <td>100.221328</td>
      <td>48.481038</td>
      <td>13.344630</td>
      <td>39.327146</td>
      <td>10.268539</td>
      <td>4.884344</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>2015-01-05 18:00:00</td>
      <td>188.576444</td>
      <td>487.336742</td>
      <td>88.967297</td>
      <td>36.571052</td>
      <td>8.278605</td>
      <td>76.534023</td>
      <td>11.892088</td>
      <td>1.945849</td>
      <td>173.315167</td>
      <td>444.049581</td>
      <td>101.633306</td>
      <td>47.279992</td>
      <td>15.793146</td>
      <td>42.984028</td>
      <td>10.006300</td>
      <td>4.637101</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>2015-01-05 21:00:00</td>
      <td>166.681364</td>
      <td>481.685320</td>
      <td>104.154110</td>
      <td>38.662638</td>
      <td>11.957697</td>
      <td>25.052743</td>
      <td>11.999161</td>
      <td>4.804263</td>
      <td>173.743459</td>
      <td>446.505202</td>
      <td>100.540356</td>
      <td>45.527290</td>
      <td>16.132288</td>
      <td>40.754154</td>
      <td>9.744855</td>
      <td>4.591048</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>291370</th>
      <td>100</td>
      <td>2015-10-02 06:00:00</td>
      <td>165.259415</td>
      <td>432.364050</td>
      <td>96.793097</td>
      <td>38.697882</td>
      <td>16.715588</td>
      <td>9.197585</td>
      <td>11.016730</td>
      <td>9.167743</td>
      <td>169.115085</td>
      <td>459.202414</td>
      <td>99.099044</td>
      <td>39.342719</td>
      <td>12.889019</td>
      <td>60.409151</td>
      <td>11.549081</td>
      <td>5.671215</td>
    </tr>
    <tr>
      <th>291371</th>
      <td>100</td>
      <td>2015-10-02 09:00:00</td>
      <td>185.907346</td>
      <td>465.062411</td>
      <td>94.161434</td>
      <td>36.156060</td>
      <td>22.822289</td>
      <td>64.351154</td>
      <td>6.469484</td>
      <td>1.656610</td>
      <td>168.838507</td>
      <td>455.101759</td>
      <td>98.960206</td>
      <td>38.277576</td>
      <td>13.917591</td>
      <td>56.810848</td>
      <td>11.118412</td>
      <td>6.061118</td>
    </tr>
    <tr>
      <th>291372</th>
      <td>100</td>
      <td>2015-10-02 12:00:00</td>
      <td>167.546991</td>
      <td>448.203119</td>
      <td>99.383591</td>
      <td>39.659572</td>
      <td>2.573507</td>
      <td>84.299208</td>
      <td>2.490792</td>
      <td>2.252574</td>
      <td>169.223690</td>
      <td>463.630715</td>
      <td>99.296474</td>
      <td>38.406915</td>
      <td>14.611939</td>
      <td>56.534099</td>
      <td>8.553177</td>
      <td>5.893650</td>
    </tr>
    <tr>
      <th>291373</th>
      <td>100</td>
      <td>2015-10-02 15:00:00</td>
      <td>175.468904</td>
      <td>441.861941</td>
      <td>105.814802</td>
      <td>38.788653</td>
      <td>9.104554</td>
      <td>48.615069</td>
      <td>6.004070</td>
      <td>3.244295</td>
      <td>172.163049</td>
      <td>458.787617</td>
      <td>100.063674</td>
      <td>38.458947</td>
      <td>15.232866</td>
      <td>49.321412</td>
      <td>8.345687</td>
      <td>5.292801</td>
    </tr>
    <tr>
      <th>291374</th>
      <td>100</td>
      <td>2015-10-02 18:00:00</td>
      <td>157.401371</td>
      <td>459.332121</td>
      <td>93.247465</td>
      <td>42.236723</td>
      <td>14.711827</td>
      <td>45.268580</td>
      <td>5.590642</td>
      <td>2.204472</td>
      <td>173.119397</td>
      <td>456.196849</td>
      <td>100.114215</td>
      <td>39.063775</td>
      <td>15.920072</td>
      <td>48.742610</td>
      <td>7.909495</td>
      <td>5.249418</td>
    </tr>
    <tr>
      <th>291375</th>
      <td>100</td>
      <td>2015-10-02 21:00:00</td>
      <td>168.651510</td>
      <td>430.056138</td>
      <td>104.487324</td>
      <td>35.735005</td>
      <td>16.328969</td>
      <td>45.108180</td>
      <td>9.103806</td>
      <td>6.093867</td>
      <td>168.410263</td>
      <td>448.519301</td>
      <td>100.049480</td>
      <td>39.083797</td>
      <td>14.635941</td>
      <td>50.313046</td>
      <td>7.582133</td>
      <td>5.108806</td>
    </tr>
    <tr>
      <th>291376</th>
      <td>100</td>
      <td>2015-10-03 00:00:00</td>
      <td>168.623762</td>
      <td>497.504580</td>
      <td>100.682235</td>
      <td>40.610939</td>
      <td>12.914771</td>
      <td>25.775781</td>
      <td>11.444951</td>
      <td>3.359673</td>
      <td>168.849711</td>
      <td>450.330382</td>
      <td>100.243957</td>
      <td>38.469840</td>
      <td>14.915283</td>
      <td>50.148100</td>
      <td>7.928488</td>
      <td>5.518228</td>
    </tr>
    <tr>
      <th>291377</th>
      <td>100</td>
      <td>2015-10-03 03:00:00</td>
      <td>168.537058</td>
      <td>441.837105</td>
      <td>87.893111</td>
      <td>40.076219</td>
      <td>23.866338</td>
      <td>36.817201</td>
      <td>16.820180</td>
      <td>0.482728</td>
      <td>171.513651</td>
      <td>452.462106</td>
      <td>98.514174</td>
      <td>38.626944</td>
      <td>14.750503</td>
      <td>44.620373</td>
      <td>8.348839</td>
      <td>5.182752</td>
    </tr>
    <tr>
      <th>291378</th>
      <td>100</td>
      <td>2015-10-03 06:00:00</td>
      <td>161.436752</td>
      <td>401.579802</td>
      <td>90.792431</td>
      <td>35.624101</td>
      <td>14.429283</td>
      <td>85.801834</td>
      <td>16.225371</td>
      <td>1.396074</td>
      <td>169.625319</td>
      <td>451.508453</td>
      <td>97.368816</td>
      <td>38.762103</td>
      <td>16.055332</td>
      <td>49.218444</td>
      <td>9.546008</td>
      <td>5.020598</td>
    </tr>
    <tr>
      <th>291379</th>
      <td>100</td>
      <td>2015-10-03 09:00:00</td>
      <td>188.559785</td>
      <td>491.929571</td>
      <td>102.821662</td>
      <td>40.034460</td>
      <td>12.668164</td>
      <td>41.768856</td>
      <td>9.448383</td>
      <td>8.454640</td>
      <td>167.950194</td>
      <td>453.659486</td>
      <td>97.292065</td>
      <td>38.635159</td>
      <td>15.942467</td>
      <td>59.763569</td>
      <td>9.431896</td>
      <td>4.476863</td>
    </tr>
    <tr>
      <th>291380</th>
      <td>100</td>
      <td>2015-10-03 12:00:00</td>
      <td>164.149847</td>
      <td>466.463280</td>
      <td>100.447025</td>
      <td>40.694147</td>
      <td>16.460088</td>
      <td>31.589198</td>
      <td>10.188202</td>
      <td>5.086589</td>
      <td>168.530246</td>
      <td>448.103742</td>
      <td>97.730594</td>
      <td>39.228047</td>
      <td>16.122100</td>
      <td>57.394487</td>
      <td>9.171658</td>
      <td>4.866485</td>
    </tr>
    <tr>
      <th>291381</th>
      <td>100</td>
      <td>2015-10-03 15:00:00</td>
      <td>184.727094</td>
      <td>487.140570</td>
      <td>100.860907</td>
      <td>36.430834</td>
      <td>12.401664</td>
      <td>57.508894</td>
      <td>18.394116</td>
      <td>7.153156</td>
      <td>169.765458</td>
      <td>458.759394</td>
      <td>98.775532</td>
      <td>38.935773</td>
      <td>16.994700</td>
      <td>55.665652</td>
      <td>9.630112</td>
      <td>4.896106</td>
    </tr>
    <tr>
      <th>291382</th>
      <td>100</td>
      <td>2015-10-03 18:00:00</td>
      <td>171.585447</td>
      <td>479.021432</td>
      <td>101.846824</td>
      <td>49.115340</td>
      <td>13.318878</td>
      <td>40.471453</td>
      <td>10.951704</td>
      <td>0.649491</td>
      <td>169.818254</td>
      <td>460.880691</td>
      <td>97.926823</td>
      <td>39.295892</td>
      <td>15.077105</td>
      <td>56.097762</td>
      <td>9.818266</td>
      <td>4.919802</td>
    </tr>
    <tr>
      <th>291383</th>
      <td>100</td>
      <td>2015-10-03 21:00:00</td>
      <td>162.144446</td>
      <td>404.865418</td>
      <td>98.384468</td>
      <td>35.389856</td>
      <td>22.516178</td>
      <td>36.216388</td>
      <td>11.492089</td>
      <td>7.269283</td>
      <td>172.199254</td>
      <td>459.599763</td>
      <td>98.365107</td>
      <td>39.964091</td>
      <td>14.414241</td>
      <td>59.222526</td>
      <td>10.569736</td>
      <td>4.693342</td>
    </tr>
    <tr>
      <th>291384</th>
      <td>100</td>
      <td>2015-10-04 00:00:00</td>
      <td>166.584930</td>
      <td>437.980304</td>
      <td>104.019479</td>
      <td>43.766793</td>
      <td>22.109109</td>
      <td>65.256390</td>
      <td>12.621841</td>
      <td>1.793100</td>
      <td>170.812061</td>
      <td>453.681180</td>
      <td>98.994157</td>
      <td>40.013985</td>
      <td>13.522301</td>
      <td>61.197614</td>
      <td>10.103349</td>
      <td>4.286568</td>
    </tr>
    <tr>
      <th>291385</th>
      <td>100</td>
      <td>2015-10-04 03:00:00</td>
      <td>173.182209</td>
      <td>452.585928</td>
      <td>106.572235</td>
      <td>40.534601</td>
      <td>16.930726</td>
      <td>38.788180</td>
      <td>10.747137</td>
      <td>6.510290</td>
      <td>170.439828</td>
      <td>455.036021</td>
      <td>99.830401</td>
      <td>40.129183</td>
      <td>14.551122</td>
      <td>65.430315</td>
      <td>9.389132</td>
      <td>4.484563</td>
    </tr>
    <tr>
      <th>291386</th>
      <td>100</td>
      <td>2015-10-04 06:00:00</td>
      <td>155.554082</td>
      <td>464.175866</td>
      <td>102.615428</td>
      <td>36.003311</td>
      <td>7.678204</td>
      <td>24.248612</td>
      <td>6.064152</td>
      <td>5.007039</td>
      <td>172.264492</td>
      <td>453.800429</td>
      <td>102.163329</td>
      <td>40.050108</td>
      <td>13.103400</td>
      <td>62.190761</td>
      <td>9.128160</td>
      <td>4.502950</td>
    </tr>
    <tr>
      <th>291387</th>
      <td>100</td>
      <td>2015-10-04 09:00:00</td>
      <td>163.814555</td>
      <td>433.614467</td>
      <td>114.798438</td>
      <td>36.454615</td>
      <td>5.259901</td>
      <td>40.947023</td>
      <td>10.677648</td>
      <td>8.252193</td>
      <td>170.243198</td>
      <td>455.333806</td>
      <td>102.290708</td>
      <td>40.197103</td>
      <td>13.120654</td>
      <td>57.910021</td>
      <td>9.238228</td>
      <td>4.803393</td>
    </tr>
    <tr>
      <th>291388</th>
      <td>100</td>
      <td>2015-10-04 12:00:00</td>
      <td>169.196188</td>
      <td>403.488184</td>
      <td>94.199431</td>
      <td>39.189491</td>
      <td>22.977467</td>
      <td>27.176467</td>
      <td>9.430194</td>
      <td>13.841831</td>
      <td>167.765844</td>
      <td>451.355029</td>
      <td>103.465520</td>
      <td>40.252524</td>
      <td>13.315211</td>
      <td>59.466979</td>
      <td>9.471171</td>
      <td>4.442337</td>
    </tr>
    <tr>
      <th>291389</th>
      <td>100</td>
      <td>2015-10-04 15:00:00</td>
      <td>165.814250</td>
      <td>446.765824</td>
      <td>99.334107</td>
      <td>44.464271</td>
      <td>1.457549</td>
      <td>58.086715</td>
      <td>1.622380</td>
      <td>3.173978</td>
      <td>167.312650</td>
      <td>439.435929</td>
      <td>102.009695</td>
      <td>40.435349</td>
      <td>11.768963</td>
      <td>60.646198</td>
      <td>9.136786</td>
      <td>5.149517</td>
    </tr>
    <tr>
      <th>291390</th>
      <td>100</td>
      <td>2015-10-04 18:00:00</td>
      <td>167.848340</td>
      <td>438.393471</td>
      <td>90.054937</td>
      <td>40.301288</td>
      <td>15.958412</td>
      <td>40.168662</td>
      <td>11.238292</td>
      <td>3.633503</td>
      <td>166.963446</td>
      <td>438.276090</td>
      <td>101.637723</td>
      <td>40.172602</td>
      <td>14.141910</td>
      <td>58.372646</td>
      <td>8.463179</td>
      <td>5.221115</td>
    </tr>
    <tr>
      <th>291391</th>
      <td>100</td>
      <td>2015-10-04 21:00:00</td>
      <td>173.508300</td>
      <td>439.917848</td>
      <td>93.063793</td>
      <td>38.750136</td>
      <td>7.633479</td>
      <td>44.399657</td>
      <td>11.019912</td>
      <td>4.952713</td>
      <td>165.979125</td>
      <td>435.138421</td>
      <td>102.005617</td>
      <td>39.497908</td>
      <td>14.000620</td>
      <td>59.820047</td>
      <td>6.856021</td>
      <td>5.392136</td>
    </tr>
    <tr>
      <th>291392</th>
      <td>100</td>
      <td>2015-10-05 00:00:00</td>
      <td>182.432617</td>
      <td>497.264899</td>
      <td>95.443869</td>
      <td>40.594815</td>
      <td>9.940475</td>
      <td>77.558997</td>
      <td>4.707020</td>
      <td>3.106529</td>
      <td>168.142646</td>
      <td>447.915202</td>
      <td>99.620102</td>
      <td>39.635003</td>
      <td>14.372707</td>
      <td>59.563942</td>
      <td>7.988174</td>
      <td>5.256284</td>
    </tr>
    <tr>
      <th>291393</th>
      <td>100</td>
      <td>2015-10-05 03:00:00</td>
      <td>158.783988</td>
      <td>438.405164</td>
      <td>100.420803</td>
      <td>40.153025</td>
      <td>10.849108</td>
      <td>72.556330</td>
      <td>2.576581</td>
      <td>4.504970</td>
      <td>168.398872</td>
      <td>448.148851</td>
      <td>99.351099</td>
      <td>39.518646</td>
      <td>15.763140</td>
      <td>57.682117</td>
      <td>8.088214</td>
      <td>5.301986</td>
    </tr>
    <tr>
      <th>291394</th>
      <td>100</td>
      <td>2015-10-05 06:00:00</td>
      <td>183.150826</td>
      <td>426.209117</td>
      <td>98.880399</td>
      <td>34.418557</td>
      <td>20.539063</td>
      <td>29.605169</td>
      <td>13.588936</td>
      <td>7.168643</td>
      <td>168.651040</td>
      <td>446.075986</td>
      <td>98.741443</td>
      <td>39.840623</td>
      <td>15.331755</td>
      <td>60.839923</td>
      <td>7.891711</td>
      <td>5.269038</td>
    </tr>
    <tr>
      <th>291395</th>
      <td>100</td>
      <td>2015-10-05 09:00:00</td>
      <td>188.267556</td>
      <td>407.256175</td>
      <td>108.931184</td>
      <td>36.553233</td>
      <td>9.599915</td>
      <td>40.722980</td>
      <td>1.639521</td>
      <td>5.724500</td>
      <td>171.826650</td>
      <td>441.278667</td>
      <td>98.311919</td>
      <td>39.196175</td>
      <td>16.429023</td>
      <td>62.147934</td>
      <td>7.475540</td>
      <td>5.448962</td>
    </tr>
    <tr>
      <th>291396</th>
      <td>100</td>
      <td>2015-10-05 12:00:00</td>
      <td>167.859576</td>
      <td>465.992407</td>
      <td>107.953155</td>
      <td>42.708899</td>
      <td>14.190347</td>
      <td>92.277799</td>
      <td>9.577243</td>
      <td>0.735339</td>
      <td>174.657123</td>
      <td>444.147310</td>
      <td>98.520388</td>
      <td>38.820190</td>
      <td>17.019808</td>
      <td>64.730136</td>
      <td>8.961444</td>
      <td>5.833191</td>
    </tr>
    <tr>
      <th>291397</th>
      <td>100</td>
      <td>2015-10-05 15:00:00</td>
      <td>170.348099</td>
      <td>434.234744</td>
      <td>104.514343</td>
      <td>38.607950</td>
      <td>10.232598</td>
      <td>49.524471</td>
      <td>12.445345</td>
      <td>2.596743</td>
      <td>173.787879</td>
      <td>448.842085</td>
      <td>100.028549</td>
      <td>39.375067</td>
      <td>17.096392</td>
      <td>64.718132</td>
      <td>9.420879</td>
      <td>5.738756</td>
    </tr>
    <tr>
      <th>291398</th>
      <td>100</td>
      <td>2015-10-05 18:00:00</td>
      <td>152.265370</td>
      <td>459.557611</td>
      <td>103.536524</td>
      <td>40.718426</td>
      <td>6.758667</td>
      <td>27.051145</td>
      <td>12.824247</td>
      <td>2.752883</td>
      <td>172.496791</td>
      <td>442.086577</td>
      <td>100.361794</td>
      <td>38.943434</td>
      <td>15.119775</td>
      <td>65.929509</td>
      <td>8.836617</td>
      <td>6.139142</td>
    </tr>
    <tr>
      <th>291399</th>
      <td>100</td>
      <td>2015-10-05 21:00:00</td>
      <td>162.887965</td>
      <td>481.415205</td>
      <td>96.687092</td>
      <td>37.162591</td>
      <td>20.541773</td>
      <td>55.057460</td>
      <td>11.713728</td>
      <td>3.539798</td>
      <td>170.782713</td>
      <td>448.188498</td>
      <td>100.794970</td>
      <td>38.980896</td>
      <td>15.573014</td>
      <td>61.859239</td>
      <td>9.942610</td>
      <td>6.191276</td>
    </tr>
  </tbody>
</table>
<p>290601 rows × 18 columns</p>
</div>




```python
final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')

print(final_feat.head())
final_feat.describe()
```

       machineID            datetime  voltmean_3h  rotatemean_3h  pressuremean_3h  \
    0          1 2015-01-02 06:00:00   180.133784     440.608320        94.137969   
    1          1 2015-01-02 09:00:00   176.364293     439.349655       101.553209   
    2          1 2015-01-02 12:00:00   160.384568     424.385316        99.598722   
    3          1 2015-01-02 15:00:00   170.472461     442.933997       102.380586   
    4          1 2015-01-02 18:00:00   163.263806     468.937558       102.726648   
    
       vibrationmean_3h  voltsd_3h  rotatesd_3h  pressuresd_3h  vibrationsd_3h  \
    0         41.551544  21.322735    48.770512       2.135684       10.037208   
    1         36.105580  18.952210    51.329636      13.789279        6.737739   
    2         36.094637  13.047080    13.702496       9.988609        1.639962   
    3         40.483002  16.642354    56.290447       3.305739        8.854145   
    4         40.921802  17.424688    38.680380       9.105775        3.060781   
    
      ...   error2count  error3count  error4count  error5count   comp1    comp2  \
    0 ...           0.0          0.0          0.0          0.0  20.000  215.000   
    1 ...           0.0          0.0          0.0          0.0  20.125  215.125   
    2 ...           0.0          0.0          0.0          0.0  20.250  215.250   
    3 ...           0.0          0.0          0.0          0.0  20.375  215.375   
    4 ...           0.0          0.0          0.0          0.0  20.500  215.500   
    
         comp3    comp4   model  age  
    0  155.000  170.000  model3   18  
    1  155.125  170.125  model3   18  
    2  155.250  170.250  model3   18  
    3  155.375  170.375  model3   18  
    4  155.500  170.500  model3   18  
    
    [5 rows x 29 columns]
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>voltmean_24h</th>
      <th>...</th>
      <th>error1count</th>
      <th>error2count</th>
      <th>error3count</th>
      <th>error4count</th>
      <th>error5count</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>...</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
      <td>290601.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.380935</td>
      <td>170.774427</td>
      <td>446.609386</td>
      <td>100.858340</td>
      <td>40.383609</td>
      <td>13.300173</td>
      <td>44.453951</td>
      <td>8.885780</td>
      <td>4.440575</td>
      <td>170.775661</td>
      <td>...</td>
      <td>0.027560</td>
      <td>0.027058</td>
      <td>0.022846</td>
      <td>0.019955</td>
      <td>0.009780</td>
      <td>53.382610</td>
      <td>51.256589</td>
      <td>52.536687</td>
      <td>53.679601</td>
      <td>11.345226</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.798424</td>
      <td>9.498824</td>
      <td>33.119738</td>
      <td>7.411701</td>
      <td>3.475512</td>
      <td>6.966389</td>
      <td>23.214291</td>
      <td>4.656364</td>
      <td>2.319989</td>
      <td>4.720237</td>
      <td>...</td>
      <td>0.166026</td>
      <td>0.164401</td>
      <td>0.151266</td>
      <td>0.140998</td>
      <td>0.098931</td>
      <td>62.478424</td>
      <td>59.156008</td>
      <td>58.822946</td>
      <td>59.658975</td>
      <td>5.826345</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>125.532506</td>
      <td>211.811184</td>
      <td>72.118639</td>
      <td>26.569635</td>
      <td>0.025509</td>
      <td>0.078991</td>
      <td>0.027417</td>
      <td>0.015278</td>
      <td>155.812721</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>164.447794</td>
      <td>427.564793</td>
      <td>96.239534</td>
      <td>38.147458</td>
      <td>8.028675</td>
      <td>26.906319</td>
      <td>5.369959</td>
      <td>2.684556</td>
      <td>168.072275</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.250000</td>
      <td>12.000000</td>
      <td>13.000000</td>
      <td>12.875000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>170.432407</td>
      <td>448.380260</td>
      <td>100.235357</td>
      <td>40.145874</td>
      <td>12.495542</td>
      <td>41.793798</td>
      <td>8.345801</td>
      <td>4.173704</td>
      <td>170.212704</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>32.625000</td>
      <td>29.500000</td>
      <td>32.125000</td>
      <td>32.375000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.000000</td>
      <td>176.610017</td>
      <td>468.443933</td>
      <td>104.406534</td>
      <td>42.226898</td>
      <td>17.688520</td>
      <td>59.092354</td>
      <td>11.789358</td>
      <td>5.898512</td>
      <td>172.462228</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>68.500000</td>
      <td>65.875000</td>
      <td>67.125000</td>
      <td>70.250000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>241.420717</td>
      <td>586.682904</td>
      <td>162.309656</td>
      <td>69.311324</td>
      <td>58.444332</td>
      <td>179.903039</td>
      <td>35.659369</td>
      <td>18.305595</td>
      <td>220.782618</td>
      <td>...</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>491.875000</td>
      <td>348.875000</td>
      <td>370.875000</td>
      <td>394.875000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>



# Label Construction
When using multi-class classification for predicting failure due to a problem, labelling is done by taking a time window prior to the failure of an asset and labelling the feature records that fall into that window as "about to fail due to a problem" while labelling all other records as "Â€Âœnormal." This time window should be picked according to the business case: in some situations it may be enough to predict failures hours in advance, while in others days or weeks may be needed to allow e.g. for arrival of replacement parts.

The prediction problem for this example scenerio is to estimate the probability that a machine will fail in the near future due to a failure of a certain component. More specifically, the goal is to compute the probability that a machine will fail in the next 24 hours due to a certain component failure (component 1, 2, 3, or 4). Below, a categorical failure feature is created to serve as the label. All records within a 24 hour window before a failure of component 1 have failure=comp1, and so on for components 2, 3, and 4; all records not within 24 hours of a component failure have failure=none.


```python
labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
labeled_features = labeled_features.fillna(method='bfill', limit=7) # fill backward up to 24h
labeled_features = labeled_features.fillna('none')
labeled_features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>...</th>
      <th>error3count</th>
      <th>error4count</th>
      <th>error5count</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
      <th>model</th>
      <th>age</th>
      <th>failure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-01-02 06:00:00</td>
      <td>180.133784</td>
      <td>440.608320</td>
      <td>94.137969</td>
      <td>41.551544</td>
      <td>21.322735</td>
      <td>48.770512</td>
      <td>2.135684</td>
      <td>10.037208</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.000</td>
      <td>215.000</td>
      <td>155.000</td>
      <td>170.000</td>
      <td>model3</td>
      <td>18</td>
      <td>none</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-01-02 09:00:00</td>
      <td>176.364293</td>
      <td>439.349655</td>
      <td>101.553209</td>
      <td>36.105580</td>
      <td>18.952210</td>
      <td>51.329636</td>
      <td>13.789279</td>
      <td>6.737739</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.125</td>
      <td>215.125</td>
      <td>155.125</td>
      <td>170.125</td>
      <td>model3</td>
      <td>18</td>
      <td>none</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-01-02 12:00:00</td>
      <td>160.384568</td>
      <td>424.385316</td>
      <td>99.598722</td>
      <td>36.094637</td>
      <td>13.047080</td>
      <td>13.702496</td>
      <td>9.988609</td>
      <td>1.639962</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.250</td>
      <td>215.250</td>
      <td>155.250</td>
      <td>170.250</td>
      <td>model3</td>
      <td>18</td>
      <td>none</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-01-02 15:00:00</td>
      <td>170.472461</td>
      <td>442.933997</td>
      <td>102.380586</td>
      <td>40.483002</td>
      <td>16.642354</td>
      <td>56.290447</td>
      <td>3.305739</td>
      <td>8.854145</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.375</td>
      <td>215.375</td>
      <td>155.375</td>
      <td>170.375</td>
      <td>model3</td>
      <td>18</td>
      <td>none</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-01-02 18:00:00</td>
      <td>163.263806</td>
      <td>468.937558</td>
      <td>102.726648</td>
      <td>40.921802</td>
      <td>17.424688</td>
      <td>38.680380</td>
      <td>9.105775</td>
      <td>3.060781</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.500</td>
      <td>215.500</td>
      <td>155.500</td>
      <td>170.500</td>
      <td>model3</td>
      <td>18</td>
      <td>none</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



Below is an example of records that are labeled as failure=comp4 in the failure column. Notice that the first 8 records all occur in the 24-hour window before the first recorded failure of component 4. The next 8 records are within the 24 hour window before another failure of component 4.


```python
labeled_features.loc[labeled_features['failure'] == 'comp4'][:16]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>machineID</th>
      <th>datetime</th>
      <th>voltmean_3h</th>
      <th>rotatemean_3h</th>
      <th>pressuremean_3h</th>
      <th>vibrationmean_3h</th>
      <th>voltsd_3h</th>
      <th>rotatesd_3h</th>
      <th>pressuresd_3h</th>
      <th>vibrationsd_3h</th>
      <th>...</th>
      <th>error3count</th>
      <th>error4count</th>
      <th>error5count</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
      <th>model</th>
      <th>age</th>
      <th>failure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2015-01-04 09:00:00</td>
      <td>166.281848</td>
      <td>453.787824</td>
      <td>106.187582</td>
      <td>51.990080</td>
      <td>24.276228</td>
      <td>23.621315</td>
      <td>11.176731</td>
      <td>3.394073</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.125</td>
      <td>217.125</td>
      <td>157.125</td>
      <td>172.125</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2015-01-04 12:00:00</td>
      <td>175.412103</td>
      <td>445.450581</td>
      <td>100.887363</td>
      <td>54.251534</td>
      <td>34.918687</td>
      <td>11.001625</td>
      <td>10.580336</td>
      <td>2.921501</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.250</td>
      <td>217.250</td>
      <td>157.250</td>
      <td>172.250</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>2015-01-04 15:00:00</td>
      <td>157.347716</td>
      <td>451.882075</td>
      <td>101.289380</td>
      <td>48.602686</td>
      <td>24.617739</td>
      <td>28.950883</td>
      <td>9.966729</td>
      <td>2.356486</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.375</td>
      <td>217.375</td>
      <td>157.375</td>
      <td>172.375</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>2015-01-04 18:00:00</td>
      <td>176.450550</td>
      <td>446.033068</td>
      <td>84.521555</td>
      <td>47.638836</td>
      <td>8.071400</td>
      <td>76.511343</td>
      <td>2.636879</td>
      <td>4.108621</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.500</td>
      <td>217.500</td>
      <td>157.500</td>
      <td>172.500</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>2015-01-04 21:00:00</td>
      <td>190.325814</td>
      <td>422.692565</td>
      <td>107.393234</td>
      <td>49.552856</td>
      <td>8.390777</td>
      <td>7.176553</td>
      <td>4.262645</td>
      <td>7.598552</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.625</td>
      <td>217.625</td>
      <td>157.625</td>
      <td>172.625</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>2015-01-05 00:00:00</td>
      <td>169.985134</td>
      <td>458.929418</td>
      <td>91.494362</td>
      <td>54.882021</td>
      <td>9.451483</td>
      <td>12.052752</td>
      <td>3.685906</td>
      <td>6.621183</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.750</td>
      <td>217.750</td>
      <td>157.750</td>
      <td>172.750</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>2015-01-05 03:00:00</td>
      <td>149.082619</td>
      <td>412.180336</td>
      <td>93.509785</td>
      <td>54.386079</td>
      <td>19.075952</td>
      <td>30.715081</td>
      <td>3.090266</td>
      <td>6.530610</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.875</td>
      <td>217.875</td>
      <td>157.875</td>
      <td>172.875</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>2015-01-05 06:00:00</td>
      <td>185.782709</td>
      <td>439.531288</td>
      <td>99.413660</td>
      <td>51.558082</td>
      <td>14.495664</td>
      <td>45.663743</td>
      <td>4.289212</td>
      <td>7.330397</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000</td>
      <td>218.000</td>
      <td>158.000</td>
      <td>0.000</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>1</td>
      <td>2015-06-18 09:00:00</td>
      <td>169.324639</td>
      <td>453.923471</td>
      <td>101.313249</td>
      <td>53.092274</td>
      <td>28.155693</td>
      <td>42.557599</td>
      <td>7.688674</td>
      <td>2.488851</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.125</td>
      <td>29.125</td>
      <td>14.125</td>
      <td>134.125</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1338</th>
      <td>1</td>
      <td>2015-06-18 12:00:00</td>
      <td>190.691297</td>
      <td>441.577271</td>
      <td>97.192512</td>
      <td>44.025425</td>
      <td>6.296827</td>
      <td>47.271008</td>
      <td>7.577957</td>
      <td>4.648336</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.250</td>
      <td>29.250</td>
      <td>14.250</td>
      <td>134.250</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1339</th>
      <td>1</td>
      <td>2015-06-18 15:00:00</td>
      <td>163.602957</td>
      <td>433.781185</td>
      <td>93.173047</td>
      <td>43.051368</td>
      <td>18.147449</td>
      <td>30.242516</td>
      <td>10.870615</td>
      <td>2.740922</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.375</td>
      <td>29.375</td>
      <td>14.375</td>
      <td>134.375</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>1</td>
      <td>2015-06-18 18:00:00</td>
      <td>178.587550</td>
      <td>427.300815</td>
      <td>118.643186</td>
      <td>50.958609</td>
      <td>2.229649</td>
      <td>17.168087</td>
      <td>15.714144</td>
      <td>5.669003</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.500</td>
      <td>29.500</td>
      <td>14.500</td>
      <td>134.500</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1341</th>
      <td>1</td>
      <td>2015-06-18 21:00:00</td>
      <td>158.851795</td>
      <td>520.113831</td>
      <td>101.974559</td>
      <td>44.156671</td>
      <td>14.554854</td>
      <td>77.101968</td>
      <td>4.788908</td>
      <td>5.468742</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.625</td>
      <td>29.625</td>
      <td>14.625</td>
      <td>134.625</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>1</td>
      <td>2015-06-19 00:00:00</td>
      <td>162.191516</td>
      <td>453.545010</td>
      <td>101.521779</td>
      <td>49.136659</td>
      <td>12.553190</td>
      <td>33.332139</td>
      <td>5.983913</td>
      <td>1.893250</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.750</td>
      <td>29.750</td>
      <td>14.750</td>
      <td>134.750</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1343</th>
      <td>1</td>
      <td>2015-06-19 03:00:00</td>
      <td>166.732741</td>
      <td>485.036994</td>
      <td>100.284288</td>
      <td>44.587560</td>
      <td>11.099161</td>
      <td>57.308864</td>
      <td>3.052958</td>
      <td>3.062215</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>89.875</td>
      <td>29.875</td>
      <td>14.875</td>
      <td>134.875</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>1</td>
      <td>2015-06-19 06:00:00</td>
      <td>172.059069</td>
      <td>463.242610</td>
      <td>96.905050</td>
      <td>53.701413</td>
      <td>14.757880</td>
      <td>55.874000</td>
      <td>3.204981</td>
      <td>2.329615</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000</td>
      <td>30.000</td>
      <td>15.000</td>
      <td>0.000</td>
      <td>model3</td>
      <td>18</td>
      <td>comp4</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 30 columns</p>
</div>



# Modelling
After the feature engineering and labelling steps, either Azure Machine Learning Studio or this notebook can be used to create a predictive model. The recommend Azure Machine Learning Studio experiment can be found in the Cortana Intelligence Gallery: Predictive Maintenance Modelling Guide Experiment. Below, we describe the modelling process and provide an example Python model.

# Training, Validation and Testing
When working with time-stamped data as in this example, record partitioning into training, validation, and test sets should be performed carefully to prevent overestimating the performance of the models. In predictive maintenance, the features are usually generated using lagging aggregates: records in the same time window will likely have identical labels and similar feature values. These correlations can give a model an "unfair advantage" when predicting on a test set record that shares its time window with a training set record. We therefore partition records into training, validation, and test sets in large chunks, to minimize the number of time intervals shared between them.

Predictive models have no advance knowledge of future chronological trends: in practice, such trends are likely to exist and to adversely impact the model's performance. To obtain an accurate assessment of a predictive model's performance, we recommend training on older records and validating/testing using newer records.

For both of these reasons, a time-dependent record splitting strategy is an excellent choice for predictive maintenace models. The split is effected by choosing a point in time based on the desired size of the training and test sets: all records before the timepoint are used for training the model, and all remaining records are used for testing. (If desired, the timeline could be further divided to create validation sets for parameter selection.) To prevent any records in the training set from sharing time windows with the records in the test set, we remove any records at the boundary -- in this case, by ignoring 24 hours' worth of data prior to the timepoint.


```python
from sklearn.ensemble import GradientBoostingClassifier

# make test and training splits
threshold_dates = [[pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
                   [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')],
                   [pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00')]]

test_results = []
models = []
for last_train_date, first_test_date in threshold_dates:
    # split out training and test data
    train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
    train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                        'machineID',
                                                                                                        'failure'], 1))
    test_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['datetime',
                                                                                                       'machineID',
                                                                                                       'failure'], 1))
    # train and predict using the model, storing results for later
    my_model = GradientBoostingClassifier(random_state=42)
    my_model.fit(train_X, train_y)
    test_result = pd.DataFrame(labeled_features.loc[labeled_features['datetime'] > first_test_date])
    test_result['predicted_failure'] = my_model.predict(test_X)
    test_results.append(test_result)
    models.append(my_model)
```


```python
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
labels, importances = zip(*sorted(zip(test_X.columns, models[0].feature_importances_), reverse=True, key=lambda x: x[1]))
plt.xticks(range(len(labels)), labels)
_, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.bar(range(len(importances)), importances)
plt.ylabel('Importance')
```




    <matplotlib.text.Text at 0x256510e55c0>




![png](output_47_1.png)


# Evaluation
In predictive maintenance, machine failures are usually rare occurrences in the lifetime of the assets compared to normal operation. This causes an imbalance in the label distribution which usually causes poor performance as algorithms tend to classify majority class examples better at the expense of minority class examples as the total misclassification error is much improved when majority class is labeled correctly. This causes low recall rates although accuracy can be high and becomes a larger problem when the cost of false alarms to the business is very high. To help with this problem, sampling techniques such as oversampling of the minority examples are usually used along with more sophisticated techniques which are not covered in this notebook.


```python
sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
labeled_features['failure'].value_counts().plot(kind='bar')
plt.xlabel('Component failing')
plt.ylabel('Count')
```




    <matplotlib.text.Text at 0x25600910f60>




![png](output_49_1.png)


Also, due to the class imbalance problem, it is important to look at evaluation metrics other than accuracy alone and compare those metrics to the baseline metrics which are computed when random chance is used to make predictions rather than a machine learning model. The comparison will bring out the value and benefits of using a machine learning model better.

In the following, we use an evaluation function that computes many important evaluation metrics along with baseline metrics for classification problems. For a detailed explanation of the metrics, please refer to the scikit-learn documentation and a companion blog post (with examples in R, not Python),


```python
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

def Evaluate(predicted, actual, labels):
    output_labels = []
    output = []
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
    print(cm)
    
    # Calculate precision, recall, and F1 score
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    precision = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])
    
    # Calculate the macro versions of these metrics
    output.extend([[np.mean(precision)] * len(labels),
                   [np.mean(recall)] * len(labels),
                   [np.mean(f1)] * len(labels)])
    output_labels.extend(['macro precision', 'macro recall', 'macro F1'])
    
    # Find the one-vs.-all confusion matrix
    cm_row_sums = cm.sum(axis = 1)
    cm_col_sums = cm.sum(axis = 0)
    s = np.zeros((2, 2))
    for i in range(len(labels)):
        v = np.array([[cm[i, i],
                       cm_row_sums[i] - cm[i, i]],
                      [cm_col_sums[i] - cm[i, i],
                       np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
        s += v
    s_row_sums = s.sum(axis = 1)
    
    # Add average accuracy and micro-averaged  precision/recall/F1
    avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
    micro_prf = [float(s[0,0]) / s_row_sums[0]] * len(labels)
    output.extend([avg_accuracy, micro_prf])
    output_labels.extend(['average accuracy',
                          'micro-averaged precision/recall/F1'])
    
    # Compute metrics for the majority classifier
    mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
    cm_row_dist = cm_row_sums / float(np.sum(cm))
    mc_accuracy = 0 * cm_row_dist; mc_accuracy[mc_index] = cm_row_dist[mc_index]
    mc_recall = 0 * cm_row_dist; mc_recall[mc_index] = 1
    mc_precision = 0 * cm_row_dist
    mc_precision[mc_index] = cm_row_dist[mc_index]
    mc_F1 = 0 * cm_row_dist;
    mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
    output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
                   mc_precision.tolist(), mc_F1.tolist()])
    output_labels.extend(['majority class accuracy', 'majority class recall',
                          'majority class precision', 'majority class F1'])
        
    # Random accuracy and kappa
    cm_col_dist = cm_col_sums / float(np.sum(cm))
    exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
    kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
    output.extend([exp_accuracy.tolist(), kappa.tolist()])
    output_labels.extend(['expected accuracy', 'kappa'])
    

    # Random guess
    rg_accuracy = np.ones(len(labels)) / float(len(labels))
    rg_precision = cm_row_dist
    rg_recall = np.ones(len(labels)) / float(len(labels))
    rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
    output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
                   rg_recall.tolist(), rg_F1.tolist()])
    output_labels.extend(['random guess accuracy', 'random guess precision',
                          'random guess recall', 'random guess F1'])
    
    # Random weighted guess
    rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist**2)
    rwg_precision = cm_row_dist
    rwg_recall = cm_row_dist
    rwg_F1 = cm_row_dist
    output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
                   rwg_recall.tolist(), rwg_F1.tolist()])
    output_labels.extend(['random weighted guess accuracy',
                          'random weighted guess precision',
                          'random weighted guess recall',
                          'random weighted guess F1'])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels
                  
    return output_df
```


```python
evaluation_results = []
for i, test_result in enumerate(test_results):
    print('\nSplit %d:' % (i+1))
    evaluation_result = Evaluate(actual = test_result['failure'],
                                 predicted = test_result['predicted_failure'],
                                 labels = ['none', 'comp1', 'comp2', 'comp3', 'comp4'])
    evaluation_results.append(evaluation_result)
evaluation_results[0]  # show full results for first split only
```

    
    Split 1:
    Confusion matrix
    - x-axis is true labels (none, comp1, etc.)
    - y-axis is predicted labels
    [[119594     21      0      4      3]
     [    18    515      3      5      1]
     [     0      0    860      0      1]
     [    13      0      2    372      1]
     [     2      2      6      0    497]]
    
    Split 2:
    Confusion matrix
    - x-axis is true labels (none, comp1, etc.)
    - y-axis is predicted labels
    [[95266    13     0     4     3]
     [   19   399     2     1     1]
     [    0     0   700     0     0]
     [   12     0     2   291     1]
     [    2     2     4     0   392]]
    
    Split 3:
    Confusion matrix
    - x-axis is true labels (none, comp1, etc.)
    - y-axis is predicted labels
    [[71724     7     0     4     3]
     [   17   300     1     1     1]
     [    0     1   547     0     0]
     [   11     0     0   212     1]
     [    2     1     3     0   274]]
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>none</th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy</th>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.999724</td>
      <td>0.957249</td>
      <td>0.987371</td>
      <td>0.976378</td>
      <td>0.988072</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.999766</td>
      <td>0.950185</td>
      <td>0.998839</td>
      <td>0.958763</td>
      <td>0.980276</td>
    </tr>
    <tr>
      <th>F1</th>
      <td>0.999745</td>
      <td>0.953704</td>
      <td>0.993072</td>
      <td>0.967490</td>
      <td>0.984158</td>
    </tr>
    <tr>
      <th>macro precision</th>
      <td>0.981759</td>
      <td>0.981759</td>
      <td>0.981759</td>
      <td>0.981759</td>
      <td>0.981759</td>
    </tr>
    <tr>
      <th>macro recall</th>
      <td>0.977566</td>
      <td>0.977566</td>
      <td>0.977566</td>
      <td>0.977566</td>
      <td>0.977566</td>
    </tr>
    <tr>
      <th>macro F1</th>
      <td>0.979634</td>
      <td>0.979634</td>
      <td>0.979634</td>
      <td>0.979634</td>
      <td>0.979634</td>
    </tr>
    <tr>
      <th>average accuracy</th>
      <td>0.999731</td>
      <td>0.999731</td>
      <td>0.999731</td>
      <td>0.999731</td>
      <td>0.999731</td>
    </tr>
    <tr>
      <th>micro-averaged precision/recall/F1</th>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
      <td>0.999327</td>
    </tr>
    <tr>
      <th>majority class accuracy</th>
      <td>0.981152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>majority class recall</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>majority class precision</th>
      <td>0.981152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>majority class F1</th>
      <td>0.990486</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>expected accuracy</th>
      <td>0.962796</td>
      <td>0.962796</td>
      <td>0.962796</td>
      <td>0.962796</td>
      <td>0.962796</td>
    </tr>
    <tr>
      <th>kappa</th>
      <td>0.981922</td>
      <td>0.981922</td>
      <td>0.981922</td>
      <td>0.981922</td>
      <td>0.981922</td>
    </tr>
    <tr>
      <th>random guess accuracy</th>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>random guess precision</th>
      <td>0.981152</td>
      <td>0.004446</td>
      <td>0.007062</td>
      <td>0.003182</td>
      <td>0.004158</td>
    </tr>
    <tr>
      <th>random guess recall</th>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>random guess F1</th>
      <td>0.332269</td>
      <td>0.008698</td>
      <td>0.013642</td>
      <td>0.006265</td>
      <td>0.008148</td>
    </tr>
    <tr>
      <th>random weighted guess accuracy</th>
      <td>0.962755</td>
      <td>0.962755</td>
      <td>0.962755</td>
      <td>0.962755</td>
      <td>0.962755</td>
    </tr>
    <tr>
      <th>random weighted guess precision</th>
      <td>0.981152</td>
      <td>0.004446</td>
      <td>0.007062</td>
      <td>0.003182</td>
      <td>0.004158</td>
    </tr>
    <tr>
      <th>random weighted guess recall</th>
      <td>0.981152</td>
      <td>0.004446</td>
      <td>0.007062</td>
      <td>0.003182</td>
      <td>0.004158</td>
    </tr>
    <tr>
      <th>random weighted guess F1</th>
      <td>0.981152</td>
      <td>0.004446</td>
      <td>0.007062</td>
      <td>0.003182</td>
      <td>0.004158</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
