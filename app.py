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
    if var1=='Male':
        data_1=0
    else:
        data_1=1
        
    var2= str(chest)
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
    if var3=='Lower than 120 mg/ml':
        data_5=0
    else:
        data_5=1
        
    var4= str(ecg)
    if var4=='ST-T wave abnormality':
        data_6=0
    elif var4=='Normal':
        data_6=1
    else:
        data_6=2
        
    data_7=int(hrt_rate)
    var5=str(exer)
    if var5=='No':
        data_8=0
    else:
        data_8=1

    data_9=float(peak)

    var6=str(slope)
    if var6=='Downsloping':
        data_10=0
    elif var6=='Flat':
        data_10=1
    else:
        data_10=2
        
    var7=str(color)
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
