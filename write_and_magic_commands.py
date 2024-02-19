



import streamlit as st

import streamlit as st

# Using object notation
contact_method = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Display content based on the selected contact method
if contact_method == "Email":
    st.write("# Hello, World1111111111111111!")
    st.write("You selected Email as the contact method.")




    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt



    st.write('Hello, *World!* :sunglasses:')




    st.write(1234)


    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40],
    }))


    st.write('1 + 1 = ', 2)

    data_frame=pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]})



    st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')



    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    c = alt.Chart(df).mark_circle().encode(
        x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

    st.write(c)



elif contact_method == "Home phone":
    st.write("# Hello, World!aaaaaaaaaaaaaaa")
    st.write("You selected Home phone as the contact method.")


    

    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd

    # Assuming 'data' is a DataFrame with 'salary' and 'test_score' columns
    data = {'salary': [50000, 60000, 75000, 90000, 80000, 70000,1222],
            'test_score': [80, 85, 88, 92, 78, 95,100]}
    df = pd.DataFrame(data)

    # Create a histogram using plotly.graph_objects with custom number of bins
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=df['test_score'], y=df['salary'], 
                            nbinsx=5,  # Adjust the number of bins for the x-axis
        
                            histfunc='sum', 
                            name='Salary vs Test Score', marker_color='blue'))

    # Update layout for better appearance
    fig.update_layout(title='Salary vs Test Score',
                    xaxis_title='Test Score',
                    yaxis_title='Salary',
                    bargap=0.1,  # Adjust the gap between bars
                    showlegend=True)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Box(x=df['test_score']))
    fig.update_layout(title='Salary vs Test Score',
                    xaxis_title='Test Score',
                    yaxis_title='Salary',
                    bargap=0.1,  # Adjust the gap between bars
                    showlegend=True   ,
                    xaxis=dict(tickangle=45)
                    )

    st.plotly_chart(fig)
elif contact_method == "Mobile phone":
    st.write("# Hello, Worldsssssssssssssssssss!")
    st.write("You selected Mobile phone as the contact method.")
