import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df  = pd.read_csv("HeartDiseaseTrain-Test.csv")

#converting the string attributes to there respective integer values
df.sex[df.sex =='Male']=0
df.sex[df.sex =='Female']=1
df.chest_pain_type[df.chest_pain_type == 'Typical angina']=0
df.chest_pain_type[df.chest_pain_type == 'Atypical angina']=1
df.chest_pain_type[df.chest_pain_type == 'Non-anginal pain']=2
df.chest_pain_type[df.chest_pain_type == 'Asymptomatic']=3
df.fasting_blood_sugar[df.fasting_blood_sugar =='Lower than 120 mg/ml']=0
df.fasting_blood_sugar[df.fasting_blood_sugar =='Greater than 120 mg/ml']=1
df.rest_ecg[df.rest_ecg =='ST-T wave abnormality']=0
df.rest_ecg[df.rest_ecg =='Normal']=1
df.rest_ecg[df.rest_ecg =='Left ventricular hypertrophy']=2
df.exercise_induced_angina[df.exercise_induced_angina =='No']=0
df.exercise_induced_angina[df.exercise_induced_angina =='Yes']=1
df.slope[df.slope =='Downsloping']=0
df.slope[df.slope =='Flat']=1
df.slope[df.slope =='Upsloping']=2
df.vessels_colored_by_flourosopy[df.vessels_colored_by_flourosopy =='Zero']=0
df.vessels_colored_by_flourosopy[df.vessels_colored_by_flourosopy =='One']=1
df.vessels_colored_by_flourosopy[df.vessels_colored_by_flourosopy =='Two']=2
df.vessels_colored_by_flourosopy[df.vessels_colored_by_flourosopy =='Three']=3
df.vessels_colored_by_flourosopy[df.vessels_colored_by_flourosopy =='Four']=4
df.thalassemia[df.thalassemia =='Reversable Defect']=0
df.thalassemia[df.thalassemia =='Normal']=1
df.thalassemia[df.thalassemia =='Fixed Defect']=2
df.thalassemia[df.thalassemia =='No']=3

#print(df)

x = df.drop(['target'],axis=1)
y=df['target']

#splitting the data into training data and test data
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

#model training
#Logistic_Regresssion
model_1 = LogisticRegression()

#loading the logistic regression model with the training data

model_1.fit(x_train,y_train)
pickle.dump(model_1, open('model.pkl','wb'))

#model evaluation
#finding accuarcy on training data
x_train_prediction = model_1.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

#finding accuarcy on training data
x_test_prediction = model_1.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

#print('Accyracy of the Training Data is :',training_data_accuracy*100,'%')
#print('Accyracy of the Testing Data is :',test_data_accuracy*100,'%')


#converting the data into the numpy arrays
#np_array = np.asarray(data)

#reshape the numpy array as predicting for only on ecolumn
"""reshaped_data = np_array.reshape(1,-1)

prediction = model.predict(reshaped_data)
print("The Predicted Value is : ",prediction)

if(prediction[0]==0):
    print("The Person does not have Heart Disease")
else:
    print("Sorry the Person have Heart Disease")"""
