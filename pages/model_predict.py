import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from my_utils import predict_sequence_class


def app():
    st.markdown("### Upload a DNA Seqence to Classify")
    st.markdown("""
    > In this section, you can enter a coding DNA sequence and our algorithm will predict to which of the classes in the table below the DNA string belongs to. 

    """)
    st.markdown("""
| Protein Class      | Name | Description
| ----------- | ----------- | ----------
| 0      | G protein coupled receptors       | [Description](https://en.wikipedia.org/wiki/G_protein-coupled_receptor)
| 1   | Tyrosine kinase        | [Description](https://en.wikipedia.org/wiki/Tyrosine_kinase)
| 2   | Tyrosine phosphatase        | [Description](https://en.wikipedia.org/wiki/Protein_tyrosine_phosphatase)
| 3   | Synthetase        | [Description](https://en.wikipedia.org/w/index.php?title=Synthetase&redirect=no)
| 4   | Synthase     | [Description](https://en.wikipedia.org/wiki/Synthase)
| 5   | Ion channel        | [Description](https://en.wikipedia.org/wiki/Ion_channel)
| 6   | Transcription factor        | [Description](https://en.wikipedia.org/wiki/Transcription_factor#:~:text=In%20molecular%20biology%2C%20a%20transcription,to%20a%20specific%20DNA%20sequence.)""")
    
    st.write(""" > Type or copy paste the DNA string into the input section below. Or you can upload a text file that contains your DNA string. You can get a sample DNA string [here](https://github.com/CyrilleMesue/dna-seq-classifier/blob/main/datasets/sample.txt)""")
    
    st.write("\n\n")
    input_text = st.text_input("Type in a coding DNA sequence", placeholder="ATCG... or atcg...")
    if input_text != "":
        call_prediction_models_on_text(input_text)
    
    st.markdown("**...OR**")
    input_file = st.file_uploader("Please upload a file containing the DNA", type=["txt"])
    if st.button("Load and Predict"):
        # set the on_click
        if input_file == None:
            st.markdown("**PLEASE UPLOAD A FILE**")
        else:
            uploaded_data = str(input_file.getvalue().decode("utf-8"))
            call_prediction_models_on_text(uploaded_data)
            
            
            
def call_prediction_models_on_text(text):
    
    class_labels = {"0" : "G protein coupled receptors", "1" : "Tyrosine kinase", "2" :"Tyrosine phosphatase", "3" : "Synthetase", "4" : "Synthase", "5" : "Ion channel", "6" : "Transcription factor"}
    
    dna_vectorizer = 'models/dna_vectorizer.sav'
    vectorizer_model = pickle.load(open(dna_vectorizer, 'rb'))
    
    dna_classifier = 'models/dna_classifier.sav'
    classifier_model = pickle.load(open(dna_classifier, 'rb'))
    predicted_class = predict_sequence_class(text, vectorizer_model, classifier_model, class_labels)
    
    output = ""
    if predicted_class[0]  in ["A", "E", "O", "U", "I"]:
        output = "Your uploaded DNA is an **"+predicted_class+"**"
    else:
        output = "Your uploaded DNA is a **"+predicted_class+"**"
        
    st.markdown(output)
        
        
    
    
    
    
