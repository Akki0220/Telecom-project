#import libraries
import streamlit as st 
import pandas as pd
import numpy as np

# header and subheader
st.header('Welcome To Telecom communication prediction')
st.subheader('Telecom data Prediction')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib 

#Read csv File 
df = pd.read_csv('make_regressor.csv')

#Feature and Target Variable
x = df.drop('satisfaction_score',axis=1)
y = df['satisfaction_score']

#Split the data into train and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

#Build a LinearRegression model
model = LinearRegression()
model.fit(x_train,y_train)

#Save the model
joblib.dump(model, 'Linear_Regression_Model.pkl')

#Load the trained model
model = joblib.load('Linear_Regression_Model.pkl')

#Predict The Satisfaction Score
def score_display():
    score = model.score(x_test,y_test)
    st.write(f'r2 score: {score:.4f}')
    
def pred(input_data):
    predict = model.predict(input_data)
    return predict 

def main():
    st.title('Score of model at test data')
    st.write('app to predict using a Linear Regression Model.')
    
    st.subheader('score of model at test data')
    score_display()
    
    st.header('input_feature')
    feature1 = st.number_input('Number_of_xDR_Sessions')
    feature2 = st.number_input('Session_Duration')
    feature3 = st.number_input('Social_Media_Total')
    feature4 = st.number_input('Google_Total')
    feature5 = st.number_input('Email_Total')
    feature6 = st.number_input('Youtube_Total')
    feature7 = st.number_input('Netflix_Total')
    feature8 = st.number_input('Gaming_Total')
    feature9 = st.number_input('Other_Total')
    feature10 = st.number_input('total_traffic')
    feature11 = st.number_input('Avg RTT DL (ms)')
    feature12= st.number_input('Avg RTT UL (ms)')
    feature13 = st.number_input('Avg Bearer TP DL (kbps)')
    feature14 = st.number_input('Avg Bearer TP UL (kbps)')
    feature15 = st.number_input('TCP DL Retrans. Vol (Bytes)')
    feature23 = st.number_input('TCP UL Retrans. Vol (Bytes)')
    feature16 = st.number_input('Activity Duration UL (ms)')
    feature17= st.number_input('Activity Duration DL (ms)')
    feature18= st.number_input('Avg RTT DL')
    feature19= st.number_input('Avg Bearer TP ')
    feature20= st.number_input('Avg TCP Retrans')
    feature21= st.number_input('Avg Activity Duration')
    feature22= st.number_input('Avg RTT')
    
    input_data = np.array([[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,
                            feature12,feature13,feature14,feature15,feature16,feature17,feature18,feature19,feature20,
                            feature21,feature22,feature23]])

    if st.button('pred'):
        predict = pred(input_data)
        st.success(f'The Predicted value is: {predict[0]:0.2f}')
        
if __name__== '__main__':
    main()
    



    
    
    
    