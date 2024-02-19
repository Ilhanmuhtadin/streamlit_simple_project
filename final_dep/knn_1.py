
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from io import StringIO
from joblib import dump, load
import io
import pickle


def knn_1_x():
    st.title('Advertising_poly')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('..\data/gene_expression.csv')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.head())

        with st.form("my_form"):
                number1 = st.number_input("age", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("physical_score", value=None,key="12", placeholder="Type a number...")

                loaded_model = load('..\model/a_4_knn\knn_model.joblib')
                

            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:



                    hasil=loaded_model.predict([[number1,number2]])
                    
                    kemungkinan=loaded_model.predict_proba([[number1,number2]])
                    
                    if hasil[0]==1:
                        st.write(hasil[0])
                        #st.write(hasil[0],'your hearing is good0',kemungkinan[0][0],kemungkinan[0][1])
                    else:
                        #st.write(hasil[0],'your hearing is not good',kemungkinan[0][0],kemungkinan[0][1])
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
            file_name='gene_expression.csv',
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
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        

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

    with tab5:
        st.header("Github")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

