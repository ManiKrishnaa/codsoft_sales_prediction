import pandas as pd

df = pd.read_csv("C:\\Users\\manik\\OneDrive\\Documents\\codsoft\\sales\\advertising.csv")
df.head()

# checking for any null values in the dataset
df.isnull().sum()  

import matplotlib.pyplot as plt

plt.scatter(df['TV'],df['Sales'])

plt.scatter(df['Radio'],df['Sales'])

plt.scatter(df['Newspaper'],df['Sales'])

# checking correlation between variables TV and Sales
from scipy.stats import pearsonr
corr , pvalue = pearsonr(df['TV'],df['Sales'])
print(" correlation between variable tv and sales : %.2f"%corr)
# hence there is strongest realtionship between variables tv and sales

# checking correlation between variables Radio and Sales
corr , pvalue = pearsonr(df['Radio'],df['Sales'])
print(" correlation between variable Radio and sales : %.2f"%corr)
# hence there is low relationship between variables radio and sales

# checking correlation between variables Newspaper and Sales
corr , pvalue = pearsonr(df['Newspaper'],df['Sales'])
print(" correlation between variable Newspaper and sales : %.2f"%corr)
# hence there is low relationship between variables Newspaper and sales

WE CAN SEE FROM THE INSIGHTS THAT THE TV HAS A BIG IMPACT ON SALES RATHER THAN RADIO AND NEWSPAPER

 from the insights of the correlations, allocating a larger portion of our advertising budget to TV, as it has a strong - impact on sales. we should consider reducing the budget for radio and newspaper advertising as their impact on sales is low

#  now we are building a model which predicting sales on tv advertising

# now first we are removing those radio,newspaper column from our data frame 
df = df.drop(df[['Radio','Newspaper']],axis=1)

df.head()

# separating our input variable and target variable lets say x and y
x = df[['TV']]
y = df[['Sales']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# now we are using a simple linear regression to predict the sales on tv advertising .
# as it has a strong linear realtionship between them
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
mse = mean_squared_error(y_test,y_pred)
print(" mean squared error : ",mse)
rmse = math.sqrt(mse)
print(" root mean square error : ",rmse)

CONCLUSION

Based on my analysis, I observed that there is a strong relationship between TV advertising and sales, as evidenced by both the high correlation coefficient and the performance of our predictive model. This insight suggests that allocating a larger portion of the advertising budget to TV advertising is likely to yield a more significant impact on sales. On the other hand, the correlation between radio and newspaper advertising budgets and sales was relatively weak, indicating that their contribution to sales prediction might be limited.
