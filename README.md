## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
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
  ```
import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
```

<img width="505" height="443" alt="image" src="https://github.com/user-attachments/assets/04c635ed-f849-4686-9358-5cb9a1f905b8" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```

<img width="430" height="290" alt="image" src="https://github.com/user-attachments/assets/b300b179-b367-4dd0-8a2d-dc782528b200" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```
<img width="562" height="437" alt="image" src="https://github.com/user-attachments/assets/cdaa96b7-c092-44a8-b29e-9b2bc81db59a" />

```
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```
<img width="566" height="437" alt="image" src="https://github.com/user-attachments/assets/eb56fc16-57e9-42a2-8f07-5ea40f2d869c" />

```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
# Orders in Alphabetical Order Blue , Green, Red 
df2=pd.concat([df2,enc],axis=1) 
df2
```
<img width="718" height="462" alt="image" src="https://github.com/user-attachments/assets/48e4c522-d3dc-40b0-bb01-627e7e8c19fd" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="1035" height="426" alt="image" src="https://github.com/user-attachments/assets/f669fc0d-c757-4cc9-8dd3-d6bbdb0f5238" />

```
import pandas as pd
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```
<img width="791" height="470" alt="image" src="https://github.com/user-attachments/assets/3b278c2c-a41d-456e-a539-df3576e81950" />

```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```
<img width="1023" height="427" alt="image" src="https://github.com/user-attachments/assets/93e51b6d-df58-4de4-b901-9fdc412427e2" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="967" height="473" alt="image" src="https://github.com/user-attachments/assets/7def5a0d-69db-4008-957f-56e77face335" />

```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="1028" height="437" alt="image" src="https://github.com/user-attachments/assets/8bfd6191-7a09-4aef-ac54-c68af7466dd8" />

```
df.skew()
```
<img width="600" height="143" alt="image" src="https://github.com/user-attachments/assets/17d24567-dfbd-4e92-9e98-e2a14f32c9cb" />
```
np.log(df["Highly Positive Skew"])
```
<img width="832" height="331" alt="image" src="https://github.com/user-attachments/assets/09f2a08a-d1e2-4ca4-a264-5cb1da66ca1e" />
```
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="887" height="335" alt="image" src="https://github.com/user-attachments/assets/e9aaa285-3e2e-40a4-b632-ce43b33b90cf" />
```
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
```
<img width="885" height="332" alt="image" src="https://github.com/user-attachments/assets/0e03a46b-cbd5-41d2-80cb-87277cd1df9f" />

```
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
```
<img width="866" height="333" alt="image" src="https://github.com/user-attachments/assets/a665861c-d102-441a-9acb-f22f21c16947" />

```
# POWER TRANSFORMATIONS 
#        BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="1046" height="380" alt="image" src="https://github.com/user-attachments/assets/085d43b4-3f9c-4b78-8355-84b6b7430248" />

```
# YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="698" height="195" alt="image" src="https://github.com/user-attachments/assets/75b5b111-4a56-4193-8648-fd882718e574" />

```
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="1042" height="347" alt="image" src="https://github.com/user-attachments/assets/72f2d658-7554-43e9-afb3-c59c196f22c1" />

```
import seaborn as sns 
import statsmodels.api as sm 
# STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
# QQ - QUANTILE QUANTILE PLOT 
plt.show()
```
<img width="1035" height="677" alt="image" src="https://github.com/user-attachments/assets/b95fd060-42e3-4bd7-8cff-616f3c2e85dc" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```
<img width="986" height="697" alt="image" src="https://github.com/user-attachments/assets/ec8311f8-3cf6-4914-b9a3-c2824684fb0f" />
```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="1015" height="691" alt="image" src="https://github.com/user-attachments/assets/54e35e7f-4903-4d4a-82d8-1e71802cadd3" />


# RESULT:

```
   Thus the Implementation of Feature Encoding and Feature Transformation executed successfully.
```

       
