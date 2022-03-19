import os
import streamlit as st
import numpy as np
from PIL import  Image
import pandas as pd
import matplotlib.pyplot as plt
from my_utils import getKmers, get_metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Custom imports 
from multipage import MultiPage
from pages import about_app, machine_learning,model_predict# import your pages here


# NOTE: This must be the first command in your app, and must be set only once
st.set_page_config(page_title="Ex-stream-ly Cool App",
	     page_icon="🧊",
	     layout="centered",
	     initial_sidebar_state="expanded",
	     menu_items={
		 'Get Help': 'https://www.extremelycoolapp.com/help',
		 'Report a bug': "https://www.extremelycoolapp.com/bug",
		 'About': "# This is a header. This is an *extremely* cool app!"
	     }
	 )
# This hides the streamlit menu icon
st.markdown(""" <style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("DNA Sequence Classification")

# Add all your application here
app.add_page("About App", about_app.app)
app.add_page("Predict DNA Sequence Class",model_predict.app)
app.add_page("Machine Learning Pipeline", machine_learning.app)

# The main app
app.run()
