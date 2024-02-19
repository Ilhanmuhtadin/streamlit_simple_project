
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from io import StringIO
from joblib import dump, load
import io
import pickle


def svm_r_1_x():
    st.title('Advertising_poly')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('..\data/a_6_svmr\cement_slump.csv')

    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.head())

        with st.form("my_form"):
                number1 = st.number_input(df.columns[0], value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input(df.columns[1], value=None,key="2", placeholder="Type a number...")
                number3 = st.number_input(df.columns[2], value=None,key="3", placeholder="Type a number...")
                number4 = st.number_input(df.columns[3], value=None,key="4", placeholder="Type a number...")
                number5 = st.number_input(df.columns[4], value=None,key="5", placeholder="Type a number...")
                number6 = st.number_input(df.columns[5], value=None,key="6", placeholder="Type a number...")
                number7 = st.number_input(df.columns[6], value=None,key="7", placeholder="Type a number...")
                number8 = st.number_input(df.columns[7], value=None,key="8", placeholder="Type a number...")
                number9 = st.number_input(df.columns[8], value=None,key="9", placeholder="Type a number...")


                loaded_model = load('..\model/a_6_svmr\svmc_model_reg.joblib')
                loaded_model_sca = load('..\model/a_6_svmr\svmc_model_reg_scaler.joblib')
            

            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:


                    h_p=loaded_model_sca.transform([[number1,number2,number3,
                                                    number4,
                                                    number5,
                                                    number6,
                                                    number7,
                                                    number8,
                                                    number9]])
                    hasil=loaded_model.predict(h_p)
                
                    
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
            file_name='cement_slump.csv',
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

