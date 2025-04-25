
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:
Read the given Data.



STEP 2:
Clean the Data Set using Data Cleaning Process.


STEP 3:
Apply Feature Encoding for the feature in the data set.



STEP 4:
Apply Feature Transformation for the feature in the data set.



STEP 5:
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

Developed by : THRISHANTH
Reg No : 212224230291

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/14d78407-90d2-433f-8b0d-18e5636035ae)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/6b01b78f-5fcf-4aaf-b275-8d999de218b5)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/3b9e2c51-215c-4c5d-8e20-fd5f2a00b61f)


```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/05696329-2154-418f-9e3d-2a14183c9124)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/8d479293-f0f9-41e2-bc53-5a3cda9ec064)



```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/22768163-823e-4f58-82b3-4a88013e7111)


```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/bf0f6e06-0e7d-468e-b846-0e680f4c7712)



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/495ae3e6-1e48-4cc2-bfc8-ceb4ae1cab97)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/e61b720a-7942-49e1-8d2a-8259579f2246)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/aa5466b4-cd99-4b44-965f-5d4d3f342d8d)


```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/40652859-4b55-48d4-9123-85279852f0aa)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/46b484ce-54f3-43be-b371-172a510d77c6)



```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/9ea298bd-b9c7-4e1e-a3d4-44d8cf30d855)


```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/af562cc1-ef66-426e-aaec-75734e2b791f)


```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/b57d72a9-7e5f-4670-a02f-0ff73104d24f)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/30db633b-2917-460d-9507-8df37e560c7d)


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/06c78725-d1d2-41ac-866d-23d00b890a47)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/19c6457e-11a5-4d97-b2ff-c59ad5d9ba87)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6e3f089f-c9d2-40c3-a25b-9dbcca2b2bf3)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/f0c798d5-3f00-456c-8bb9-3822a476ab24)



```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/3945deb5-3e8d-4aaf-ac3c-2a35398fc6cd)



```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2b488911-00cb-4c76-bf2f-76950ff40c0b)


```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/af17cebf-e434-4a81-b9e6-0f71a03d1024)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/f65a4260-15c8-4325-8f1e-c3ddfa1c709b)




## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
