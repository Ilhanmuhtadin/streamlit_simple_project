import streamlit as st
import pandas as pd



from io import StringIO
from joblib import dump, load
import io
import pickle





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
        df=pd.read_csv('data/Advertising.csv')
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
                        
                        loaded_model = load('model/1simpl.joblib')
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
                file_name='Advertising.csv',
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
    

        with tab4:
            st.header("Accuracy")
            st.header("MAE : 0.45172351786524734")
            st.header("MSE : 0.27913788532955025")
            st.header("RMSE : 0.5283350124017433")     

        with tab5:
            st.header("Github")



elif contact_method == "Polynomial Regresion":
    st.title('Advertising poly')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/Advertising.csv')
    with tab1:
        st.header("Predict Data")

        with st.form("my_form"):
                number1 = st.number_input("TV", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("radio", value=None,key="12", placeholder="Type a number...")
                number3 = st.number_input("newspaper", value=None,key="1W", placeholder="Type a number...")

            
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:
                    
  

                    loaded_poly = load('model/a_2_poly/poly_converter.joblib')
                    loaded_model = load('model/a_2_poly/sales_poly_model.joblib')

                    campaign_poly = loaded_poly.transform([[number1,number2,number3]])
                    hasil=loaded_model.predict(campaign_poly)
                    
                    
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
            file_name='Advertising.csv',
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
        
        



    with tab4:
        st.header("Accuracy")
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        


    with tab5:
        st.header("Github")



elif contact_method == "Logistic_Regression":
    st.title('hearing test')

  


elif contact_method == "Logistic_Regression_multy_1":
    st.title('iris')




elif contact_method == "KNN Classifier":

    st.title('gene expression')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/gene_expression.csv')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.head())

        with st.form("my_form"):
                number1 = st.number_input("age", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("physical_score", value=None,key="12", placeholder="Type a number...")

                loaded_model = load('../model/a_4_knn/knn_model.joblib')
                

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
        
        


    with tab4:
        st.header("Accuracy")
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        

    with tab5:
        st.header("Github")
       

elif contact_method == "SVM Classifier":

    st.title('Mouse Viral Study')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_5_svmc\mouse_viral_study.csv')

    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.head())

        with st.form("my_form"):
                number1 = st.number_input(df.columns[0], value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input(df.columns[1], value=None,key="12", placeholder="Type a number...")

                loaded_model = load('model/a_5_svmc\svmc_model.joblib')
                loaded_model_sca = load('model/a_5_svmc\svmc_model_sca.joblib')
                
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:


                    h_p=loaded_model_sca.transform([[number1,number2]])
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
            file_name='mouse_viral_study.csv',
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
        
        




    with tab4:
        st.header("Accuracy")
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        



    with tab5:
        st.header("Github")


elif contact_method == "SVM Regression":

    st.title('cement slump')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_6_svmr\cement_slump.csv')

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


                loaded_model = load('model/a_6_svmr\svmc_model_reg.joblib')
                loaded_model_sca = load('model/a_6_svmr\svmc_model_reg_scaler.joblib')
            

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


    with tab4:
        st.header("Accuracy")
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        


    with tab5:
        st.header("Github")



elif contact_method == "random forest classifier":

    st.title('data banknote authentication')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_8_rfc\data_banknote_authentication.csv')

    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=12))




        with st.form("my_forms"):
                number1 = st.number_input(df.columns[0], value=None,key="0", placeholder="Type a number...")
                number2 = st.number_input(df.columns[1], value=None,key="1", placeholder="Type a number...")
                number3 = st.number_input(df.columns[2], value=None,key="2", placeholder="Type a number...")
                number4 = st.number_input(df.columns[3], value=None,key="3", placeholder="Type a number...")

                loaded_model = load('model/a_8_rfc\halo_rfc.joblib')

                
    
    
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:

                    hasil=loaded_model.predict([[number1,number2,number3,number4]])
                
                    
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
            file_name='data_banknote_authentication.csv',
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
        
        

    with tab4:
        st.header("Accuracy")
        st.header("MAE : 0.3926093765986013")
        st.header("MSE : 0.2578347048485534")
        st.header("RMSE : 0.5077742656422768")
        



    with tab5:
        st.header("Github")
