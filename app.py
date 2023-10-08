import streamlit as st
import pandas as pd
import numpy as np

st.title('Hello world from streamlit!')

register, verify, log = st.tabs(['Register', 'Verify', 'Log'])

with register :
    st.header('Take a nice pic and input your name to register! 😊')
    reg_picture = st.camera_input('Take a pic!', key='reg_pic')
    name_input = st.text_input('Enter your name here: ')
    if name_input :
        st.write('Great! Your name is: ', name_input)

    st.write("...And here's your nice pic!")
    if reg_picture :
        st.image(reg_picture)

    st.button('Done', type='primary')

with verify :
    st.header('Please take a pic to verify!')
    verify_picture = st.camera_input('Take a pic!', key='verify_pic')
    

with log :
    st.header('This is the log tab!')