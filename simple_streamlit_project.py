import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from io import StringIO
from joblib import dump, load
import io
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error


def dowload_job(loaded_model):
    st.header('Model data')
    buffer = io.BytesIO()
    #st.write(buffer)
    dump(loaded_model, buffer)
    buffer.seek(0)

    
    st.download_button(
        label="Download Model",
        data=buffer.getvalue(),
        file_name="model.joblib"
    )


def tabb_4(hilangkan,test_size,random_state):

    X = df.drop(hilangkan,axis=1)
    y = df[hilangkan]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    test_predictions =loaded_poly.predict(X_test)
    MAE = mean_absolute_error(y_test,test_predictions)
    MSE = mean_squared_error(y_test,test_predictions)
    RMSE = np.sqrt(MSE)

    

    mae_s="MAE : "
    mae_s1="MSE : "
    mae_s2="RMSE : "
    
    sss = mae_s + str(MAE)
    sss1 = mae_s1 + str(MSE)
    sss2 = mae_s2 + str(RMSE)


    
    st.header("Accuracy")
    st.subheader(sss)
    st.subheader(sss1)
    st.subheader(sss2)
    st.subheader("")
    st.subheader("")
    st.header("plot line prediction results")
    x_line=np.arange(len(y_test))
    df_1=pd.DataFrame(test_predictions)
    df_2=pd.DataFrame(y_test)
    df_2=df_2.reset_index(drop=True)
    df_3=pd.concat([df_2,df_1],axis=1)
    df_3=df_3.sort_values('sales')
    
    fig = plt.figure(figsize=(7,3),dpi=500)
    axes = fig.add_axes([0, 0, 1, 1]) 
    
    axes.plot(x_line,df_3[hilangkan].values,label='test')
    axes.plot(x_line,df_3[0].values,label='pred')
    plt.legend()
    st.pyplot(fig)
    st.subheader("")



# Using object notation
contact_method = st.sidebar.selectbox(
    "simple Machine Learning",
    ("linear regresion","Polynomial Regresion",'Logistic_Regression','Logistic_Regression_multy_1',"KNN Classifier","SVM Classifier",
     "SVM Regression","decision tree classifier","random forest classifier")
)





if contact_method == "linear regresion":
        st.title('Advertising s')
        
        tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                              "Accuracy","Github"])
        df=pd.read_csv('data/Advertising.csv')
        loaded_model = load('model/1simpl.joblib')
        with tab1:
            st.header("Predict Data")
            with st.expander("Sample data"):
                st.dataframe(df.sample(5,random_state=1))

            with st.form("my_form_linear regresion_Advertising"):
                    number1 = st.number_input("TV", value=None,key="linear regresion_Advertising1", placeholder="Type a number...")
                    number2 = st.number_input("radio", value=None,key="linear regresion_Advertising2", placeholder="Type a number...")
                    number3 = st.number_input("newspaper", value=None,key="linear regresion_Advertising3", placeholder="Type a number...")

                
                # Every form must have a submit button.
                    submitted = st.form_submit_button("Submits")
                    if submitted:
                        
                        
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
            st.title('Model data')
            buffer = io.BytesIO()
            #st.write(buffer)
            dump(loaded_model, buffer)
            buffer.seek(0)
        
            
            st.download_button(
                label="Download Model",
                data=buffer.getvalue(),
                file_name="model.joblib"
            )

        with tab5:
            st.header("Github")



elif contact_method == "Polynomial Regresion":
    st.title('Advertising poly')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/Advertising.csv')
    loaded_poly = load('model/a_2_poly/final_sales_poly_model_pipes.joblib')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            random_number = st.empty()
            data_button = st.button("change data")
            if data_button:
                random_number = random.randint(1, 1000)
                st.dataframe(df.sample(5, random_state=random_number))
            



        with st.form("my_form"):
                number1 = st.number_input("TV", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("radio", value=None,key="12", placeholder="Type a number...")
                number3 = st.number_input("newspaper", value=None,key="1W", placeholder="Type a number...")

            
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:
                    
  

                    
                    
                    campaign_poly = loaded_poly.predict([[number1,number2,number3]])
  
                    
                    
                    st.write(campaign_poly[0])



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
        tabb_4("sales",0.3,101)
        dowload_job(loaded_poly)
    
        


    with tab5:
        st.header("Github")
        st.write("https://github.com/Ilhanmuhtadin/deploy_simple_ml/tree/main/deploy_github")
       



elif contact_method == "Logistic_Regression":
    st.title('hearing test')


    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/hearing_test.csv')
    loaded_model = load('model/a_3_logis/model_baru.joblib')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=1))

        with st.form("my_form"):
                number1 = st.number_input("age", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("physical_score", value=None,key="12", placeholder="Type a number...")

                
                
            
            
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:


                    
                    hasil=loaded_model.predict([[number1,number2]])
                    
                    
                    if hasil[0]==1:
                        st.write(hasil[0],'your hearing is good')
                    else:
                        st.write(hasil[0],'your hearing is not good')



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
            file_name='hearing_test.csv',
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
        
        dowload_job(loaded_model)
        


      
    with tab5:
        st.header("Github")
  


elif contact_method == "Logistic_Regression_multy_1":
    st.title('iris')


    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/iris.csv')
    loaded_model = load('model/a_3_logis/model_baru1.joblib')
    
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=1))

        with st.form("my_form"):
                number1 = st.number_input("sepal_length", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("sepal_width", value=None,key="12", placeholder="Type a number...")
                number3 = st.number_input("petal_length", value=None,key="123", placeholder="Type a number...")
                number4 = st.number_input("petal_width", value=None,key="1234", placeholder="Type a number...")
                
                
            
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:
                    

                    
                    hasil=loaded_model.predict([[number1,number2,number3,number4]])
              
                    
                   

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
            file_name='iris.csv',
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
        dowload_job(loaded_model)
        

   

    with tab5:
        st.header("Github")





elif contact_method == "KNN Classifier":

    st.title('gene expression')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/gene_expression.csv')
    loaded_modelknn = load('model/a_4_knn/knn_model.joblib')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=1))

        with st.form("my_form"):
                number1 = st.number_input("age", value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input("physical_score", value=None,key="12", placeholder="Type a number...")

                


                

            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:



                    hasil=loaded_modelknn.predict([[number1,number2]])
                    
                    kemungkinan=loaded_modelknn.predict_proba([[number1,number2]])
                    
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
        buffer = io.BytesIO()
        #st.write(buffer)
        dump(loaded_modelknn, buffer)
        buffer.seek(0)
        st.header('Model data')
    
        
        st.download_button(
            label="Download Model",
            data=buffer.getvalue(),
            file_name="model.joblib"
        )
        

    with tab5:
        st.header("Github")
       

elif contact_method == "SVM Classifier":

    st.title('Mouse Viral Study')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_5_svmc/mouse_viral_study.csv')
    loaded_model = load('model/a_5_svmc/model_baru1.joblib')

    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=1))

        with st.form("my_form"):
                number1 = st.number_input(df.columns[0], value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input(df.columns[1], value=None,key="12", placeholder="Type a number...")

                
                
                
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:


                    
                    hasil=loaded_model.predict([[number1,number2]])
                
                    
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
        dowload_job(loaded_model )



    with tab5:
        st.header("Github")


elif contact_method == "SVM Regression":

    st.title('cement slump')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_6_svmr/cement_slump.csv')

    loaded_model = load('model/a_6_svmr/model_baru_1.joblib')
               
    

    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=1))

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


                loaded_model = load('model/a_6_svmr/svmc_model_reg.joblib')
               
            

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
        dowload_job(loaded_model)


    with tab5:
        st.header("Github")



elif contact_method == "random forest classifier":

    st.title('data banknote authentication')

    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_8_rfc/data_banknote_authentication.csv')
    loaded_model = load('model/a_8_rfc/halo_rfc.joblib')
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=12))




        with st.form("my_forms"):
                number1 = st.number_input(df.columns[0], value=None,key="0", placeholder="Type a number...")
                number2 = st.number_input(df.columns[1], value=None,key="1", placeholder="Type a number...")
                number3 = st.number_input(df.columns[2], value=None,key="2", placeholder="Type a number...")
                number4 = st.number_input(df.columns[3], value=None,key="3", placeholder="Type a number...")

                

                
    
    
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
        buffer = io.BytesIO()
        #st.write(buffer)
        st.header('Model data')
        dump(loaded_model, buffer)
        buffer.seek(0)
    
        
        st.download_button(
            label="Download Model",
            data=buffer.getvalue(),
            file_name="model.joblib"
        )



    with tab5:
        st.header("Github")


elif contact_method == "decision tree classifier":
    st.title('pinguin_species')
    
    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Predict Data","Simple Info Data","Distribution Data",
                                            "Accuracy","Github"])
    df=pd.read_csv('data/a_7_dtc/pinguin_ukuran.csv')
    df_kolums=pd.read_csv('data/a_7_dtc/kolums.csv')
    loaded_model = load('model/a_7_dtc/dtc_sc_com.joblib')
    
    with tab1:
        st.header("Predict Data")
        with st.expander("Sample data"):
            st.dataframe(df.sample(5,random_state=42))
    
        with st.form("my_form"):
                option1 = st.selectbox(
                'sex',
                ('MALE', 'FEMALE'))
    
                option2 = st.selectbox(
                'from island',
                ('Torgersen', 'Biscoe', 'Dream'))
                number1 = st.number_input(df.columns[2], value=None,key="1", placeholder="Type a number...")
                number2 = st.number_input(df.columns[3], value=None,key="2", placeholder="Type a number...")
                number3 = st.number_input(df.columns[4], value=None,key="3", placeholder="Type a number...")
                number4 = st.number_input(df.columns[5], value=None,key="4", placeholder="Type a number...")
    
    
                
    
                lop=df.drop(columns='species')
                
                lop1=pd.DataFrame([[option2,number1,number2,number3,number4,option1]])
                lop1.columns=lop.columns
                df_pred=pd.concat([lop1,lop])
                df_pred_d=pd.get_dummies(df_pred)
                
    
                
                #st.dataframe(df_pred_d)
                kolom=df_kolums.columns
                #st.dataframe(kolom[1:])
                df_final=df_pred_d[kolom[1:]]
                ff=df_final.iloc[[0]]
    
    
    
    
                
            # Every form must have a submit button.
                submitted = st.form_submit_button("Submits")
                if submitted:
    
    
                    
                    hasil=loaded_model.predict(ff)
                    #st.dataframe(ff)
                   
                    
                    if hasil[0]==1:
                        st.write(hasil[0])
                        #st.write(hasil[0],'your hearing is good0',kemungkinan[0][0],kemungkinan[0][1])
                    else:
                        #st.write(hasil[0],'your hearing is not good',kemungkinan[0][0],kemungkinan[0][1])
                        st.write(hasil[0])
    
    
    
    with tab2:
        st.header("Simple Info Data")
        
    
        st.dataframe(df.sample(5,random_state=42))
    
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
    
    
        
        csv = convert_df(df)
    
    
        st.download_button(
            label="Download full data as CSV",
            data=csv,
            file_name='pinguin_species.csv',
            mime='text/csv',
        )
    
    
    
        
    
        st.header("describe data")
        st.dataframe(df.describe())
        st.dataframe(df.describe(include="O"))
        
    
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
        buffer = io.BytesIO()
        st.header('Model data')
        #st.write(buffer)
        dump(loaded_model, buffer)
        buffer.seek(0)
    
        
        st.download_button(
            label="Download Model",
            data=buffer.getvalue(),
            file_name="model.joblib"
        )
        
    
       
    with tab5:
        st.header("Github")
      
    

