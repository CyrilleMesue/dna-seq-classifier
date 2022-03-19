import streamlit as st
from PIL import Image

def app():
    st.markdown("## About App")
    st.markdown("""
This app is a deployment of a DNA classification mini project using machine learning. A multinomial naive Bayes classifier is trained on the dataset and used to classifiy coding DNA sequences. 


The dataset for this study was gotten from kaggle and can be accessed through the link below:  
https://www.kaggle.com/nageshsingh/dna-sequence-dataset

And this code was adapted from : https://github.com/krishnaik06/DNA-Sequencing-Classifier

The model have been trained and validated using a 10-fold cross validation and achieved an average accuracy of 98.2 on the human dataset. These models (countvectoriizer and multinomial naive Bayes classifier) can be used to make predictions on new datasets. All that is needed is to enter a DNA text string or upload a text file in the **Predict DNA class** on the taskbar."""
               )


    image = Image.open('predict.png')
    st.image(image, caption='Upload a DNA Seqence to Classify')

    st.markdown("""
You can also train on your own dataset provided the data is in text format where the DNA(label = sequence) is seprated from the labels(label = class) by a comma. You can play with the **Machine Learning Pipeline** on the taskbar for this. It might take a few minutes to completely run the code in the background.   

> Note: the models are saved in the tmp folder of the software. If you want to customize the software it is best you download it along with the jupyter notebook provide [here]().  
    
    """)
    
    image = Image.open('machinelearning.png')
    st.image(image, caption='Machine Learning for DNA Classification')
    
    st.markdown("Github : [https://github.com/CyrilleMesue/dna-seq-classifier](https://github.com/CyrilleMesue/dna-seq-classifier)")
    
    st.markdown("***ENJOY THE APP***")
