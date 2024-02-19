import requirements
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from joblib import dump, load
import io
import pickle


import Polynomial_Regresion1
import Logistic_Regression1
import Logistic_Regression_multy_1
import knn_1
import svm_c_1
import svm_r_1
import rf_c_1


# Using object notation
contact_method = st.sidebar.selectbox(
    "simple Machine Learning",
    ("linear regresion","Polynomial Regresion",'Logistic_Regression','Logistic_Regression_multy_1',"KNN Classifier","SVM Classifier",
     "SVM Regression","random forest classifier")
)





if contact_method == "linear regresion":
        st.title('Advertising')
        
        tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                              "Accuracy","Github"])
        df=pd.read_csv('..\data/Advertising.CSV')
        with tab1:
            st.header("Predict Data")
            with st.expander("Sample data"):
                st.dataframe(df.head())

            with st.form("my_form_linear regresion_Advertising"):
                    number1 = st.number_input("TV", value=None,key="linear regresion_Advertising1", placeholder="Type a number...")
                    number2 = st.number_input("radio", value=None,key="linear regresion_Advertising2", placeholder="Type a number...")
                    number3 = st.number_input("newspaper", value=None,key="linear regresion_Advertising3", placeholder="Type a number...")

                
                # Every form must have a submit button.
                    submitted = st.form_submit_button("Submits")
                    if submitted:
                        
                        loaded_model = load('..\model/1simpl.joblib')
                        hasil=loaded_model.predict([[number1,number2,number3]])
                        st.write(hasil[0])



        with tab2:
            st.header("Simple Info Data")
            

            st.dataframe(df.head())


            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')


            
            csv = convert_df(df)

    
            st.download_button(
                label="Download full data as CSV",
                data=csv,
                file_name='Advertising.CSV',
                mime='text/csv',
            )
                

            st.header("describe data")
            st.dataframe(df.describe())

            st.header("info data")
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()

            st.text(s)



        with tab3:
            st.header("Distribution Data")
           
            
    

            with open("sks.png", "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="flower.png",
                    mime="image/png"
                )

        with tab4:
            st.header("Accuracy")
            st.header("MAE : 0.45172351786524734")
            st.header("MSE : 0.27913788532955025")
            st.header("RMSE : 0.5283350124017433")
            

            # Load the model
            loaded_model = load('..\model/1simpl.joblib')

            # Convert the model to bytes
            model_bytes = io.BytesIO()
            

            # Create the download button
            st.download_button(
                label="Download model as simpl.joblib",
                data=model_bytes,
                file_name='simpl.joblib',
                mime='application/octet-stream',  # Use the appropriate MIME type


            )

            # Load your trained model (replace with your actual model)
            loaded_models = pickle.load(open('..\model/simpl.pkl', 'rb'))
            model_bytes = io.BytesIO()
            
            

        with tab5:
            st.header("Github")
            st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


elif contact_method == "Polynomial Regresion":
     Polynomial_Regresion1.xapi()

elif contact_method == "Logistic_Regression":
     Logistic_Regression1.Logistic_Regression1_x()   

elif contact_method == "Logistic_Regression_multy_1":
     Logistic_Regression_multy_1.Logistic_Regression_multy_1_x()  


elif contact_method == "KNN Classifier":
     knn_1.knn_1_x() 

elif contact_method == "SVM Classifier":
     svm_c_1.svm_c_1_x() 
     
elif contact_method == "SVM Regression":
     svm_r_1.svm_r_1_x()

elif contact_method == "random forest classifier":
     rf_c_1.rf_c_1_x() 
