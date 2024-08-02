#import necessary libraries
import streamlit as st
import pandas as pd

#App title
st.title(" WorldBank Renewable Energy App")

#creating a paragraph
st.write(''' 
         Renewable energy can help countries mitigate climate change,
          build resilience to volatile prices, and lower energy costs.''')

#importing data to the web app
df = pd.read_csv("C:\Users\Admin\Desktop\Renewable-Energy-Consumption\WorldBank_Renewable_Energy_Internal_Project (1).ipynb")

st.write(df.head(5)) # checking the first 5 rows


#User Slider
num_rows =st.slider("Select the number of rows",min_value =1,max_value =len(df),value=5)
st.write("Here are the rows you have selected in the Dataset")
st.write(df.head(num_rows))

#finding the shape of the dataset
st.write("viewing the number Of rows and columns in the dataset:",df.shape)

# CHECK Box that can be used for duplicates if any
if st.checkbox("check duplicates"):
    st.write(df[df.duplicated()])

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Encode categorical variables
encoded_columns = ['Country Code', 'Country Name', 'Income Group', 'Indicator Code', 'Indicator Name', 'Region', 'Year']
le_dict = {col: LabelEncoder() for col in encoded_columns}

for column in encoded_columns:
    le_dict[column].fit(df[column])
    df[column] = le_dict[column].transform(df[column])

# prepare features and target variable
x=df.drop('Energy Consump.',axis=1)
y=df['Energy Consump.']
    
df.drop('Indicator Code',axis=1,inplace=True)# droping Indictor code
df.drop('Indicator Name',axis=1,inplace=True)



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)# split and training

#view shape of train and test
st.write('y_train.shape',y_train.shape)
st.write('y_test.shape',y_test.shape)
st.write('x_train.shape',x_train.shape) # You had a typo here, it should be x_train
st.write('x_test.shape',x_test.shape)

#fit the model

rfr=RandomForestRegressor(n_estimators=100,random_state=0)
rfr.fit(x_train,y_train)

#print accuracy
from sklearn.metrics import r2_score
y_pred=rfr.predict(x_test)
r2_accuracy = r2_score(y_test,y_pred)
st.write(" Accuracy score is:", r2_accuracy)

# user input for new data
st.sidebar.write("Enter New data for Prediction")

Country_Code= st.sidebar.selectbox("Country Code", le_dict['Country Code'].classes_)
Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)
Income_Group=st.sidebar.selectbox("Income Group", le_dict['Income Group'].classes_)
Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region= st.sidebar.selectbox("Region", le_dict['Region'].classes_)
Year = st.sidebar.number_input("Year")

# Encode user input
encoded_input = {
    'Country Code':le_dict['Country Code'].transform([Country_Code])[0],
    'Country Name':le_dict['Country Name'].transform([Country_Name])[0],
    'Income Group':le_dict['Income Group'].transform([Income_Group])[0],
    'Indicator Code':le_dict['Indicator Code'].transform([Indicator_Code])[0],
    'Indicator Name':le_dict['Indicator Name'].transform([Indicator_Name])[0],
    'Region':le_dict['Region'].transform([Region])[0],
    'Year': Year

}

# Convert to DataFrame

encoded_input_df=pd.DataFrame([encoded_input])

# predict using the model
if st.sidebar.button('Predict Energy Consumption'):
    prediction=rfr.predict(encoded_input_df)[0]
    st.sidebar.write('Predicted Energy consumption:',prediction)