import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# This function computes the accuracy, precision, recall and f1-score of the model
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

# This function predicts the class of any raw DNA string
def predict_sequence_class(sequence, vectorizer_model, classifier_model, class_labels):
    """
    inputs: DNA sequence, model and class labels (dictionary)
    Function takes in a DNA sequence returns its class amongst:
    [G protein coupled receptors', Tyrosine kinase', 'Tyrosine phosphatase', 'Synthetase', 'Synthase', 'Ion channel', Transcription factor'}
    
    """
    seq_o_kmers = getKmers(sequence)
    
    text_data = ' '.join(seq_o_kmers)

    input_x = vectorizer_model.transform([text_data])
    
    predicted_value = classifier_model.predict(input_x)[0]
    
    predicted_label = class_labels[str(predicted_value)]
    
    return predicted_label
