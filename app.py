import pickle
from flask import Flask,render_template,jsonify,request

app=Flask(__name__)

scaler = pickle.load(open("models/scaler_diabetes_prediction.pkl",'rb'))
classifier = pickle.load(open("models/classifier_logistic_regression_diabetes_prediction.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Temperature'))
        Glucose = float(request.form.get('RH'))
        BloodPressure = float(request.form.get('Ws'))
        SkinThickness = float(request.form.get('Rain'))
        Insulin = float(request.form.get('FFMC'))
        BMI = float(request.form.get('DMC'))
        DiabetesPedigreeFunction = float(request.form.get('ISI'))
        Age = float(request.form.get('Classes'))

        new_data_scaled=scaler.transform([['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
        result=classifier.predict(new_data_scaled)

        return render_template('home.html',result=result[0])        
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)