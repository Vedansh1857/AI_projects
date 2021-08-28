from attr import attributes
from numpy.lib.function_base import corrcoef
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image

st.write('''#Diabetes Detection#
    Detect if someone has diabetes or not using Machine Learning & Python
''')
img = Image.open('D:\\Machine learning with python\\Robotics-1.jpg')
st.image(image=img,caption='Machine Learning',use_column_width=True)

# Displaying the data in a tabular format
df = pd.read_csv('D:\Machine learning with python\diabetes.csv')
st.subheader('Data Information : ')
st.dataframe(data=df)

# Show statistics on the data :
st.write(df.describe())

# To show the data as a chart:
st.bar_chart(df)

# Separating the features & labels
x = df.iloc[:,0:8].values
y = df.iloc[:,-1].values

# Splitting the dataset into training & testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Taking the user input
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0,17,3)
    glucose = st.sidebar.slider('glucose', 0,199,117)
    blood_Pressure = st.sidebar.slider('blood_pressure', 0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0,99,23)
    insulin = st.sidebar.slider('insulin', 0.0,846.0,36.0)
    bmi = st.sidebar.slider('BMI', 0.0,67.1,32.0)
    dpf = st.sidebar.slider('DPF', 0.078,2.42,0.3725)
    age = st.sidebar.slider('age', 0,15,3)

    # Storing a dictionary into a variable
    user_data = {'pregnancies':pregnancies,
                    'glucose':glucose,
                    'blood_Pressure':blood_Pressure,
                    'skin_thickness':skin_thickness,
                    'insulin':insulin,
                    'BMI':bmi,
                    'DPF':dpf,
                    'age':age}

    # Transforming it into dataframe                    
    features = pd.DataFrame(user_data, index=[0])
    return features

# To store the user input into a variable
user_input = get_user_input()

# Setting a subheader & displaying the input
st.subheader('user input:')
st.write(user_input)

# Training & testing the model
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# Displaying the model metrics
preds = rfc.predict(x_test)
st.subheader('The model accuracy is : ')
st.write(str(accuracy_score(y_test, preds)*100)+'%')

predictions = rfc.predict(user_input)
st.subheader('Classification : ')
st.write(predictions)