from flask import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key="private_key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/registrationform')
def registrationform():
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
    model = LogisticRegression()

    #loading the logistic regression model with the training data

    model.fit(x_train,y_train)

    #model evaluation
    #finding accuarcy on training data
    x_train_prediction = model.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)

    #finding accuarcy on training data
    x_test_prediction = model.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)

    print('Accyracy of the Training Data is :',training_data_accuracy*100,'%')
    print('Accyracy of the Testing Data is :',test_data_accuracy*100,'%')

    #Building the predictive systems
    data=[None]*13

    #Taking inputs from the users
    print("Enter the values Of :")
    data[0]=int(input('age : '))
    var1=input('sex : ')
    if var1=='Male':
        data[1]=0
    else:
        data[1]=1
        
    var2=input('chest_pain_type : ')
    if var2=='Typical angina':
        data[2]=0
    elif var2=='Atypical angina':
        data[2]=1
    elif var2=='Non-anginal pain':
        data[2]=2
    else:
        data[2]=3
        
    data[3]=int(input('resting_blood_pressure : '))
    data[4]=int(input('cholestoral : '))

    var3=input('fasting_blood_sugar : ')
    if var3=='Lower than 120 mg/ml':
        data[5]=0
    else:
        data[5]=1
        
    var4=input('rest_ecg : ')
    if var4=='ST-T wave abnormality':
        data[6]=0
    elif var4=='Normal':
        data[6]=1
    else:
        data[6]=2
        
    data[7]=int(input('Max_heart_rate : '))
    var5=input('exercise_induced_angina : ')
    if var5=='No':
        data[8]=0
    else:
        data[8]=1

    data[9]=float(input('oldpeak : '))

    var6=input('slope')
    if var6=='Downsloping':
        data[10]=0
    elif var6=='Flat':
        data[10]=1
    else:
        data[10]=2
        
    var7=input('vessels_colored_by_flourosopy : ')
    if var7=='Zero':
        data[11]=0
    elif var7=='One':
        data[11]=1
    elif var7=='Two':
        data[11]=2
    elif var7=='Three':
        data[11]=3
    else:
        data[11]=4
        
    var8=input('thalassemia : ')
    if var8=='Reversable Defect':
        data[12]=0
    elif var8=='Normal':
        data[12]=1
    elif var8=='Fixed Defect':
        data[12]=2
    else:
        data[12]=3

    #converting the data into the numpy arrays
    np_array = np.asarray(data)

    #reshape the numpy array as predicting for only on ecolumn
    reshaped_data = np_array.reshape(1,-1)

    prediction = model.predict(reshaped_data)
    print("The Predicted Value is : ",prediction)

    if(prediction[0]==0):
        print("The Person does not have Heart Disease")
    else:
        print("Sorry the Person have Heart Disease")


if __name__ == "__main__":
    app.run(debug=True)
