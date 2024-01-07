import streamlit as st
from PIL import Image
import yfinance as yf
from datetime import datetime

# st.write("""
#
# # This is Test Of **Streamlit**
#
# """)
#
#
# st.sidebar.header('This is Header')
#
# st.sidebar.slider('This is slider')
# st.sidebar.text_input('Input Text','Default')
#
#
# st.header('This is Header Test')
#
# st.subheader('This is SubHeader')
#
# st.success('Success')
# st.warning('Warning')
# st.error('This is Error')
#
# img = Image.open('C:/Users/moham/OneDrive/دسکتاپ/python-stock/1.jpg')
# st.image(img,width=300,caption='Caption Test')
#
# # vid=open('Example Adderes')
# # st.video(vid)
#
# st.checkbox('Show')
# st.checkbox('hide')
#
# st.selectbox('Your Selections',['Python','Java','C++'])



st.write('''
# Finance
This is **Stock** Visualizer
''')
img = Image.open('C:/Users/moham/OneDrive/دسکتاپ/python-stock/1.jpg')
st.image(img,caption='Caption Test')

symbol = 'GOOG'
data = yf.Ticker(symbol)
Dt = datetime.now()
history = data.history(period='1d',start='2023-02-01',end=Dt.strftime('%Y-%m-%d'))
st.header('PRICE')
st.line_chart(history.Close)
st.header('Volume')
st.line_chart(history.Volume)
