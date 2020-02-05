import pandas as pd
from  matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_excel("Diagnoses of FLD.xls")
df["Age"] = df["Age"].astype(int)
x = df.drop(["Abdominal ultrasound diagnosis of FLD","Number"],axis=1)
y = df["Abdominal ultrasound diagnosis of FLD"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

model = linear_model.LogisticRegression(C=3,intercept_scaling=7,warm_start=True)
model = model.fit(x_train,y_train)
model = model.score(x_test,y_test)
print(model)

#SCORE = 0.815043156596794

"""
model = linear_model.LogisticRegression()
num = []
for i in range(1,200):
    num.append(i)

param = {"warm_start":[True,False]}
grid = GridSearchCV(model,param,cv=5)
grid = grid.fit(x_train,y_train)
print(grid.best_params_)
"""
