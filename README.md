# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 
 

## Program:
```
Developed by:Vaishak.M
RegisterNumber: 212224040355
/*
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
Developed by: Infant Maria Stefanie .F 
RegisterNumber: 212224230095
*/
```

## Output:

# TOP 5 ELEMENTS

![Ex 5 image 1](https://github.com/user-attachments/assets/7077a0b2-c1cb-4eea-b375-bbf70003954f)

![Ex 5 image 2](https://github.com/user-attachments/assets/7a8a8b25-3747-48f3-bc12-8f08d510f5ed)

![Ex 5 image 3](https://github.com/user-attachments/assets/5e090584-0735-4527-85fa-64f842e58e11)

# DATA DUPLICATE

![image](https://github.com/user-attachments/assets/ade46e8a-30a1-49e4-97e1-b3dc17eee333)

# PRINT DATA

![image](https://github.com/user-attachments/assets/17663cfc-8462-4935-bc9b-ed807900d01e)

# DATA STATUS

![image](https://github.com/user-attachments/assets/448ebc94-67ca-4b3e-9414-7927d94d011b)

# Y-PREDICTION ARRAY

![image](https://github.com/user-attachments/assets/a3045a06-c041-40db-b5b8-30ed5948ca28)

# CONFUSSION ARRAY

![image](https://github.com/user-attachments/assets/c477e140-3248-4f24-9fac-47954187700e)

# ACCURACY LEVEL

![image](https://github.com/user-attachments/assets/d5ad083f-e7d7-48c7-9db7-681248335081)

# CLASSIFICATION REPORT

![image](https://github.com/user-attachments/assets/2dc0a7d2-fc03-4372-bb1f-0c0fbe3820e8)

# PREDICTION OF LR

![image](https://github.com/user-attachments/assets/4a70e9f2-5b57-4775-87d1-d108d0da3f5f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
