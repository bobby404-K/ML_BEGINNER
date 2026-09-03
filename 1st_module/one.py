import pandas as pd

data= pd.read_csv("D:\screenshot\iris - iris.csv")

data

data.shape

data.head(5) #think like snake head

data.head(-5)

data.tail(-10) #think like snake tail

data.columns #names of the colum.

data['Id']

data['Id'].shape

data.info()

data.dtypes

data['age'].mean()


data['Id'].mean()

import matplotlib.pyplot as plt

plt.hist(data['age'], bins=7)
plt.xlabel('age')
plt.ylabel('sex')

plt.scatter(data['trestbps'], data['sex'], data['age'], data['target'])




