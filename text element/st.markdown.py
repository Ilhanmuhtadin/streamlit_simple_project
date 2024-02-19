import streamlit as st


baris_baru="== <br>=="
st.write('''<br>
         
         halo''')
st.write("ke 1 bold")
st.markdown("*Streamlit* is **really** ***cool***.")


st.write("ke 2 warna")
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors].''')


st.write("ke 3 bentuk")
st.markdown("Here's a bouquet &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")



st.write("ke 4 baris baru")
multi = '''If you end a line with two spaces,
a soft return is used for the next line.


Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)






st.write("ke 5 text area")
md = st.text_area('Type in your markdown string (without outer quotes)',
                  "Happy Streamlit-ing! :balloon:")


st.write("ke 6 kodingan")
st.code(f"""
import streamlit as st

st.markdown('''{md}''')
""")

st.markdown(md)

