# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/air-quality-data-in-india'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px #graphing
import plotly.graph_objects as go #graphing
from plotly.subplots import make_subplots #graphing
import matplotlib.pyplot as plt #graphing
import seaborn as sns #graphing
import missingno as msno #describe data
import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
# pd.set_option('max_columns', 200)

df = pd.read_csv('/kaggle/input/air-quality-data-in-india/stations.csv')

df.head()

df.shape

df['State'].value_counts()

state_stations = df.groupby('State')['StationId'].unique().reset_index()
state_stations

wb_stations = state_stations[-1:]['StationId']
print(wb_stations)

df1 = pd.read_csv('/kaggle/input/air-quality-data-in-india/station_hour.csv',low_memory=False)

df1

df1.describe()

df1.columns

df1.isna().sum()

df1['AQI_Bucket'].value_counts()

df1.dtypes

df1['Datetime'] = pd.to_datetime(df1['Datetime'])

df1.dtypes

df1.loc[df1.duplicated()]

target_station = 'WB013'
station_df = df1[df1['StationId'] == target_station]

station_df

ax = sns.scatterplot(x='Datetime',
                y='AQI',
                data=station_df)
ax.set_title('AQI over time')
plt.legend(loc='best') 
plt.xlim(df1['Datetime'].min(), df1['Datetime'].max() + pd.Timedelta(weeks=1))  # You can adjust the timedelta as needed
plt.show()

sns.displot(df1['PM2.5'],aspect=3,height=5)

df2 = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')

df2.head()

df2.dtypes

df2.describe()

df2['Date'] = pd.to_datetime(df2['Date'])

df2['City'].value_counts()

target_city = 'Patna'
City_df2 = df2[df2['City'] == target_city]

ax = sns.scatterplot(x='Date',
                y='AQI',
                hue= 'AQI',
                data=City_df2
                    )
ax.set_title('AQI over time')
# plt.legend(loc='best') 
plt.show()

plt.plot(City_df2['AQI'])
plt.xlabel('Date')
plt.ylabel('AQI')

City_df2.columns

sns.pairplot(City_df2,
             vars=['Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'],)
#             hue='Type_Main')
plt.show()

df_corr = City_df2[['Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].dropna().corr()
df_corr

sns.heatmap(df_corr, annot=True)

City_df2['month'] = City_df2['Date'].dt.month
City_df2['day'] = City_df2['Date'].dt.day

fig = px.scatter_3d(City_df2, x = "month", y = "day", z = "AQI",
                    color = "PM2.5", color_continuous_scale = ["#00FF00", "#FFC800", "#FF0000", "#B803BF"], 
                    range_color = (-45, 225)) # range of color bar

fig.update_traces(marker = dict(size = 2.5)) # scaling down the markers
fig.update_layout(template = "plotly_dark", font = dict(family = "PT Sans", size = 12))
fig.show()

fig = px.scatter_3d(City_df2, x = "Date", y = "AQI", z = "PM2.5",
                    color = "PM2.5", color_continuous_scale = ["#00FF00", "#FFC800", "#FF0000", "#B803BF"], 
                    range_color = (-45, 225)) # range of color bar

fig.update_traces(marker = dict(size = 2.5)) # scaling down the markers
fig.update_layout(template = "plotly_dark", font = dict(family = "PT Sans", size = 12))
fig.show()

df3 = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_hour.csv')

df3.head()

df3['Datetime'] = pd.to_datetime(df3['Datetime'])

df3['City'].value_counts()

target_city = 'Hyderabad'
City_df3 = df3[df3['City'] == target_city]

City_df3.head()

ax = sns.scatterplot(x='Datetime',
                y='AQI',
                hue= 'AQI',
                data=City_df3
                    )
ax.set_title('AQI over time')
# plt.legend(loc='best') 
plt.show()

plt.plot(City_df3['AQI'])
plt.xlabel('Date')
plt.ylabel('AQI')

sns.pairplot(City_df3,
             vars=['Datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'],)
#             hue='Type_Main')
plt.show()

df_corr_2 = City_df3[['Datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].dropna().corr()
df_corr_2

rounded_df_corr_2 = np.around(df_corr_2, decimals=1)

sns.heatmap(rounded_df_corr_2, annot=True)

City_df3['month'] = City_df3['Datetime'].dt.month

City_df3['hour'] = City_df3['Datetime'].dt.hour

City_df3.shape

fig = px.scatter_3d(City_df3, x = "month", y = "PM2.5", z = "AQI",
                    color = "PM2.5", color_continuous_scale = ["#00FF00", "#FFC800", "#FF0000", "#B803BF"], 
                    range_color = (-45, 225)) # range of color bar

fig.update_traces(marker = dict(size = 2.5)) # scaling down the markers
fig.update_layout(template = "plotly_dark", font = dict(family = "PT Sans", size = 12))
fig.show()

fig = px.scatter_3d(City_df3, x = "hour", y = "month", z = "PM2.5",
                    color = "PM2.5", color_continuous_scale = ["#00FF00", "#FFC800", "#FF0000", "#B803BF"], 
                    range_color = (-45, 225)) # range of color bar

fig.update_traces(marker = dict(size = 2.5)) # scaling down the markers
fig.update_layout(template = "plotly_dark", font = dict(family = "PT Sans", size = 12))
fig.show()

df4 = pd.read_csv('/kaggle/input/air-quality-data-in-india/station_day.csv')

df4.head()


