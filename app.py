from flask import *
import numpy as np
import pickle
import heart_disease

app = Flask(__name__)
app.secret_key="private_key"
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/model', methods=['GET','POST'])
def model():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    chest = request.form['chest']
    bp = request.form['bp']
    chol = request.form['chol']
    sugar = request.form['sugar']
    ecg = request.form['ecg']
    hrt_rate = request.form['hrt_rate']
    exer = request.form['exer']
    peak = request.form['peak']
    slope = request.form['slope']
    color = request.form['color']
    thala = request.form['thala']


    data_1=0
    data_2=0
    data_3=0
    data_4=0
    data_5=0
    data_6=0
    data_7=0
    data_8=0
    data_9=0
    data_10=0
    data_11=0
    data_12=0

    #Taking inputs from the users
    #print("Enter the values Of :")
    data_0 = int(age)
    var1 = str(sex)
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

    print('Accuracy of the Training Data is :',training_data_accuracy*100,'%')
    print('Accuracy of the Testing Data is :',test_data_accuracy*100,'%')

    #Building the predictive systems
    data=[None]*13

    #Taking inputs from the users
    print("Enter the values Of :")
    data[0]=int(input('Age : '))
    var1=input('Sex : (Male/Female)').lower()
    if var1=='Male':
        data_1=0
    else:
        data_1=1
        

    var2= str(chest)

    var2=input('Chest Pain Type : (Typical angina/Atypical angina/Non-anginal pain/other) ').lower()

    if var2=='Typical angina':
        data_2=0
    elif var2=='Atypical angina':
        data_2=1
    elif var2=='Non-anginal pain':
        data_2=2
    else:
        data_2=3
        

    data_3= int(bp)
    data_4= int(chol)

    var3= str(sugar)

    data[3]=int(input('Resting Blood Pressure : '))
    data[4]=int(input('Cholestoral : '))

    var3=input('Fasting Blood Sugar : ')

    if var3=='Lower than 120 mg/ml':
        data_5=0
    else:
        data_5=1
        

    var4= str(ecg)
    if var4=='ST-T wave abnormality':
        data_6=0
    elif var4=='Normal':
        data_6=1

    var4=input('Electrocardiogram : (ST-T wave abnormality/Normal/other)').lower()
    if var4=='st-t wave abnormality':
        data[6]=0
    elif var4=='normal':
        data[6]=1
    else:
        data_6=2
        
    data_7=int(hrt_rate)
    var5=str(exer)

    data[7]=int(input('Maximum heart rate : '))
    var5=input('Exercise induced angina :(No/Yes)').lower()
    if var5=='No':
        data_8=0
    else:
        data_8=1

    data_9=float(peak)

    var6=str(slope)

    data[9]=float(input('Oldpeak : '))

    var6=input('slope : (Downsloping/Flat)')
    if var6=='Downsloping':
        data_10=0
    elif var6=='Flat':
        data_10=1
    else:
        data_10=2
        
    var7=str(color)

    var7=input('Vessels Colored by flourosopy : (zero/one/two/three)')
    if var7=='Zero':
        data_11=0
    elif var7=='One':
        data_11=1
    elif var7=='Two':
        data_11=2
    elif var7=='Three':
        data_11=3
    else:
        data_11=4
        
    var8=str(thala)

    var8=input('Thalassemia : (Reversable Defect/Normal/Fixed Defect)')
    if var8=='Reversable Defect':
        data_12=0
    elif var8=='Normal':
        data_12=1
    elif var8=='Fixed Defect':
        data_12=2
    else:
        data_12=3

    final_features  = np.array([(data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12)])
    prediction = model.predict(final_features)
    return render_template("predict.html", prediction_text = "The Patient has Heart_disease : {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
