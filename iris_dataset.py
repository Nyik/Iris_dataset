import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
import altair as alt
from PIL import Image
import os
import pickle 

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
***
""")

image = Image.open('img/flowers.jpeg')

st.image(image, use_column_width=True)

st.write("""
*The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis*

*Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.*  

For more information [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set).

You will now get to see which features get which flower(Versicolor,Setosa,Virginica) with the sliders in the sidebar 
""")

st.sidebar.header('User Input Parameters')
st.sidebar.markdown('Play around with the features and see which flower you get')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv("csv/Iris.csv")

def changeNum(row):
    if row == 'Iris-setosa':
        return 1
    elif row == 'Iris-versicolor':
        return 0
    else:
        return 2

data['Species_Target'] = data['Species'].apply(changeNum)



# st.subheader('EDA(Exploratory Data Analysis)')

# for i in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
#     f, ax = plt.subplots(figsize=(7, 5))
#     # ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     ax = sns.swarmplot(y=i, x="Species", data=data)
#     st.pyplot(f)

# f, ax = plt.subplots(figsize=(7, 5))
# ax =plt.scatter(data['PetalLengthCm'], data['PetalWidthCm'], c=data['Species_Target'])
# plt.xlabel('Sepal Length', fontsize=18)
# plt.ylabel('Sepal Width', fontsize=18)
# plt.legend()
# st.pyplot(f)

# corrmat = data.corr()
# f, ax = plt.subplots(figsize=(7, 5))
# ax = sns.heatmap(corrmat, annot = True, vmax=1, square=True)
# st.pyplot(f)

X = data.drop(columns=['Id','Species_Target','Species'])
Y = data.Species_Target
# st.write(X)
# clf = RandomForestClassifier()
# clf.fit(X, Y)
load_clf = pickle.load(open('iris_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(data.Species)

st.subheader('Prediction')
iris_species = np.array(['Versicolor','Setosa','Virginica'])
# st.write(iris_species[prediction])

# st.write(iris_species[prediction][0])

image = Image.open(f'img/{iris_species[prediction][0]}.jpeg')

st.image(image)


st.subheader('Prediction Probability')
# st.write(prediction_proba)

st.write('There is  ' + str(int(prediction_proba[0][0]*100)) + "% chance the flower is Versicolor")
st.write('There are  ' + str(int(prediction_proba[0][1]*100)) + '% chance the flower is Setosa')
st.write('There are  ' + str(int(prediction_proba[0][2]*100)) + '% chance the flower is Virginica')




