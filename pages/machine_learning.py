# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_utils import getKmers, get_metrics, predict_sequence_class
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle

def app():
    st.markdown("### Machine Learning for DNA Classification")
    st.markdown("[Get Jupyter Notebook](https://raw.githubusercontent.com/CyrilleMesue/dna-seq-classifier/main/DNA%20Sequencing%20and%20applying%20Classifier.ipynb)")
    st.markdown("""
    In this notebook, we will apply a classification model that can predict a gene's function based on the DNA sequence of the coding sequence alone. The model can be used to deterimine if the gene product is any of the following classes of proteins. 

| Protein Class      | Name | Description
| ----------- | ----------- | ----------
| 0      | G protein coupled receptors       | [Description](https://en.wikipedia.org/wiki/G_protein-coupled_receptor)
| 1   | Tyrosine kinase        | [Description](https://en.wikipedia.org/wiki/Tyrosine_kinase)
| 2   | Tyrosine phosphatase        | [Description](https://en.wikipedia.org/wiki/Protein_tyrosine_phosphatase)
| 3   | Synthetase        | [Description](https://en.wikipedia.org/w/index.php?title=Synthetase&redirect=no)
| 4   | Synthase     | [Description](https://en.wikipedia.org/wiki/Synthase)
| 5   | Ion channel        | [Description](https://en.wikipedia.org/wiki/Ion_channel)
| 6   | Transcription factor        | [Description](https://en.wikipedia.org/wiki/Transcription_factor#:~:text=In%20molecular%20biology%2C%20a%20transcription,to%20a%20specific%20DNA%20sequence.)
       
       

The dataset for this study was gotten from kaggle and can be accessed through the link below:  
https://www.kaggle.com/nageshsingh/dna-sequence-dataset

And this code was adapted from : https://github.com/krishnaik06/DNA-Sequencing-Classifier
"""
    )
    st.markdown("### Upload Data")
    st.markdown("You can get  a sample dataset [here]().")
    input_file = st.file_uploader("Please upload a comma separated text file!!! First column should contain dna strings and second column, the numeric labels", type=["txt"])

    if st.button("Upload file"):
        # set the on_click
        if input_file == None:
            st.markdown("**Please upload a comma separated text file!!! First column should contain dna strings and second column, the numeric labels**")
        else:
            uploaded_data = pd.read_table(input_file)
            st.markdown("""
            ***Data Uploaded***
            """)
            
            st.dataframe(uploaded_data)
            
            st.write("There are {} data samples\n".format(len(uploaded_data)))
            
        st.markdown("""### Methodology  

If all DNA molecules were of equal lengths, then perhaps the best solution would have to convert each DNA sequence into an image and train convolutional neural networks on such images. Every other machine learning appoach is not effective due to its requirements of fixed length elements. A solution to this is to split the DNA molecules into fixed length shorter DNA strings called k-mers. Therefore, each DNA molecule will have a particular number of different k-length sub DNA strings which encode a particular information. The k-mers of a given DNA string are then considered as words and the DNA itself as a sentence or paragraph, and a natural language processing algorithm is applied. 
            """)
        
        st.markdown("### Import Utilites")
        st.code("""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_utils import getKmers, get_metrics, predict_sequence_class
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
        
        """, language='python')
        st.markdown("### Helper functions")
        st.code("""
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
    '''
    inputs: DNA sequence, model and class labels (dictionary)
    Function takes in a DNA sequence returns its class amongst:
    [G protein coupled receptors', Tyrosine kinase', 'Tyrosine phosphatase', 'Synthetase', 'Synthase', 'Ion channel', Transcription factor'}
    
    '''
    seq_o_kmers = getKmers(sequence)
    
    text_data = ' '.join(seq_o_kmers)

    input_x = vectorizer_model.transform([text_data])
    
    predicted_value = classifier_model.predict(input_x)[0]
    
    predicted_label = class_labels[str(predicted_value)]
    
    return predicted_label
        
        """, language='python')
        st.markdown("### Convert DNA strings into k-mers")
    
        st.code("""   
data_preprocessed = uploaded_data     
data_preprocessed['words'] = data_preprocessed.apply(lambda x: getKmers(x['sequence']), axis=1)       
data_preprocessed = data_preprocessed.drop('sequence', axis=1)      
st.dataframe(data_preprocessed) """, language='python'
               )

        data_preprocessed = uploaded_data
        data_preprocessed['words'] = data_preprocessed.apply(lambda x: getKmers(x['sequence']), axis=1)
        data_preprocessed = data_preprocessed.drop('sequence', axis=1)
        st.dataframe(data_preprocessed)
            
        st.markdown("""
            ### Convert list of k_mers to a string of words
This is necessing for applying Natural Language Processing Algorithms. The labels also need to be extracted from the pandas table into an array. """)
        st.code("""
# Convert to text
texts = list(data_preprocessed['words'])
for item in range(len(texts)):
    texts[item] = ' '.join(texts[item])

# extract labels
y_data = data_preprocessed.iloc[:, 0].values
texts[0]
        """, language='python'
                   )
       
        # Convert to text
        texts = list(data_preprocessed['words'])
        for item in range(len(texts)):
            texts[item] = ' '.join(texts[item])

        # extract labels
        y_data = data_preprocessed.iloc[:, 0].values

        st.markdown(texts[0])
            
        st.markdown("### Applying the BAG of WORDS using CountVectorizer using NLP")
        st.code("""
# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
cv = CountVectorizer(ngram_range=(4,4))

# This will create an n by m compressed matrix where n is the number of samples and m is the number of unique k-mers over
# all the samples
X = cv.fit_transform(texts)
X.shape""", language='python'
                   )
        # Creating the Bag of Words model using CountVectorizer()
        # This is equivalent to k-mer counting
        cv = CountVectorizer(ngram_range=(4,4))

        # This will create an n by m compressed matrix where n is the number of samples and m is the number of unique k-mers over
        # all the samples
        X = cv.fit_transform(texts)
        st.write(X.shape)

        st.markdown("### Checking for Class Imbalance dataset.")
        st.markdown("""```
data_preprocessed['class'].value_counts().sort_index().plot.bar()        
        """)
        st.bar_chart(data=data_preprocessed['class'].value_counts().sort_index(), width=800, height=500, use_container_width=False)
        st.markdown("### Data Splitting into Training and Testing")
        st.markdown("By default we use 80 percent data for training and 20 percent for testing by setting the test_size to 0.2. You can change the test_size value and try different split ratios.")

        # Splitting the human dataset into the training set and test set
        st.code("""
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.20, random_state=42)
print("Training set :  {}".format(X_train.shape))
print("Test set :  {}".format(X_test.shape))""", language='python'
                   )
        X_train, X_test, y_train, y_test = train_test_split(X,y_data,test_size = 0.20,random_state=42)
        st.write("Training set :  {}".format(X_train.shape))
        st.write("Test set :  {}".format(X_test.shape))

        st.markdown("### Training A multinomial naive Bayes classifier ")
        st.code("""
### Multinomial Naive Bayes Classifier ###
# The alpha parameter was determined by grid search previously
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)""", language='python'
                   )
        ### Multinomial Naive Bayes Classifier ###
        # The alpha parameter was determined by grid search previously
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        st.markdown("### Testing the model on the Test Data")
        st.code("""
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = {:.3f}\n precision = {:.3f} \n recall = {:.3f} \n f1 = {:.3f}".format(accuracy, precision, recall, f1))""", language='python'
                   )
        st.write("Confusion matrix\n")
        st.write(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
        accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
        st.write("accuracy = {:.3f}".format(accuracy))
        st.write("precision = {:.3f}".format(precision))
        st.write("recall = {:.3f}".format(recall))
        st.write("f1 = {:.3f}".format(f1))

        st.markdown("""
### Hyperparameter Tunning and Cross Validation
Cross validation employs that the model is trained on all the data and tested on all the data. First, the dataset is split into k equal groups. The first k-1 groups are used for the training while the k-th group is used for testing. This is reapeated over all grups such that each group becomes the test set and the average test accuracies over all groups is taken. This approach of validation provides a more convenient accuracy matrix than just randomly secting training and test set. 

Sklearn employs gridsearch to tune hyper parameters and return the various n-fold (n can be changed) cross validation accuracy for each parameter combination. Our parameter of interest is alpha."""
                   )
        
        st.code("""
clf = GridSearchCV(MultinomialNB(), {'alpha': [1,0.1,0.01,0.001,0.0001,0.00001]}, cv=10, return_train_score=True)
clf.fit(X, y_data)
df = pd.DataFrame(clf.cv_results_)
df[['param_alpha','mean_test_score', 'mean_train_score']]
print(("The best alpha value is {} ".format(clf.best_params_["alpha"]))""", language='python'
                   )
        
        clf = GridSearchCV(MultinomialNB(), {
                    'alpha': [1,0.1,0.01,0.001,0.0001,0.00001]
                }, cv=10, return_train_score=True)
        clf.fit(X, y_data)
        df = pd.DataFrame(clf.cv_results_)
        st.dataframe(df[['param_alpha','mean_test_score', 'mean_train_score']])

        st.write("The best alpha value is {} ".format(clf.best_params_["alpha"]))

        st.markdown("### Train Model on Optimal Parameter")
        st.code("""
a = clf.best_params_["alpha"]
classifier = MultinomialNB(alpha=a)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = {:.3f}\n precision = {:.3f} \n recall = {:.3f} \n f1 = {:.3f}".format(accuracy, precision, recall, f1))""", language='python'
                   )
        a = clf.best_params_["alpha"]
        classifier = MultinomialNB(alpha=a)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        st.write("Confusion matrix\n")
        st.write(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
        accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
        st.write("accuracy = {:.3f}".format(accuracy))
        st.write("precision = {:.3f}".format(precision))
        st.write("recall = {:.3f}".format(recall))
        st.write("f1 = {:.3f}".format(f1))


        st.markdown("### Stratified k-fold Cross Validation")
        st.code("""
# 10-fold cross validation Can play with n_splits parameter
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
# Print output
scores = cross_val_score(MultinomialNB(alpha=a), X, y_data, cv=kf, n_jobs=None, scoring='f1_micro')
print(f'K-Fold test: {scores}')
print(f'Mean: {scores.mean().round(3)}')
print(f'Std: {scores.std().round(3)}')""", language='python'
                   )
        # 10-fold cross validation Can play with n_splits parameter
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
        
        # Print output
        scores = cross_val_score(MultinomialNB(alpha=a), X, y_data, cv=kf, n_jobs=None, scoring='f1_micro')
        st.write(f'K-Fold test: {scores}')
        st.write(f'Mean: {scores.mean().round(3)}')
        st.write(f'Std: {scores.std().round(3)}')

        st.markdown("### Save Model")
        st.markdown("""```
# file name
filename = 'tmp/dna_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))"""
                   )
        # file name
        filename = 'tmp/dna_classifier.sav'
        pickle.dump(classifier, open(filename, 'wb'))

        st.markdown("### Load and Test Model")
        st.code("""
# load the model from disk
dna_classifier = 'tmp/dna_classifier.sav'
loaded_model = pickle.load(open(dna_classifier, 'rb'))

dna_vectorizer = 'tmp/dna_vectorizer.sav'
loaded_vectorizer = pickle.load(open(dna_vectorizer, 'rb'))

result = loaded_model.score(X_test, y_test)
print(result)

loaded_model.predict(X_test[24])
#### This is the Label Map
class_labels = {"0" : "G protein coupled receptors", "1" : "Tyrosine kinase", "2" :"Tyrosine phosphatase", "3" : "Synthetase", "4" : "Synthase", "5" : "Ion channel", "6" : "Transcription factor"}

sequence = uploaded_data.iloc[70].sequence
print(sequence)

# Predict class
predict_sequence_class(sequence, cv, loaded_model, class_labels)
""", language='python'
               )
        # load the model from disk
        dna_classifier = 'tmp/dna_classifier.sav'
        loaded_model = pickle.load(open(dna_classifier, 'rb'))

        dna_vectorizer = 'tmp/dna_vectorizer.sav'
        loaded_vectorizer = pickle.load(open(dna_vectorizer, 'rb'))

        result = loaded_model.score(X_test, y_test)
        st.write(result)

        loaded_model.predict(X_test[24])
        #### This is the Label Map
        class_labels = {"0" : "G protein coupled receptors", "1" : "Tyrosine kinase", "2" :"Tyrosine phosphatase", "3" : "Synthetase", "4" : "Synthase", "5" : "Ion channel", "6" : "Transcription factor"}

        sequence = uploaded_data.iloc[70].sequence
        #sequence

        # Predict class
        st.write(predict_sequence_class(sequence, cv, loaded_model, class_labels))

        st.markdown("### Performing Hyper parameter Tunning for both CountVectorizer and Multinomial Naive Bayes Classifier")
        st.code("""
# ngram_range values to test 
ngram_range_params = [(1,1), (1,2), (1,3), (1,4), (2,2),(2,3),(2,4),(3,3),(3,4),(4,4)]

# alpa values to test
alpha_params = [1,0.1,0.01,0.001,0.0001,0.00001]

# initialize dictionary of best parameters
best_parameters = {"best_alpha" : None, "best_ngram_range" : None}
best_score = 0

for i in range(len(ngram_range_params)):
    ngram_range = ngram_range_params[i]
    cv = CountVectorizer(ngram_range=ngram_range)
    x = cv.fit_transform(texts)
    clf = GridSearchCV(MultinomialNB(), {'alpha': alpha_params}, cv = 5, return_train_score=False)
    clf.fit(x, y_data)
    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_parameters["best_alpha"] = clf.best_params_["alpha"]
        best_parameters["best_ngram_range"] = ngram_range_params[i]

pd.DataFrame(best_parameters)""", language='python'
                   )
        # ngram_range values to test 
        ngram_range_params = [(1,1), (1,2), (1,3), (1,4), (2,2),(2,3),(2,4),(3,3),(3,4),(4,4)]

        # alpa values to test
        alpha_params = [1,0.1,0.01,0.001,0.0001,0.00001]

        # initialize dictionary of best parameters
        best_parameters = {"best_alpha" : None, "best_ngram_range" : None}
        best_score = 0

        for i in range(len(ngram_range_params)):
            ngram_range = ngram_range_params[i]
            cv = CountVectorizer(ngram_range=ngram_range)
            x = cv.fit_transform(texts)
            clf = GridSearchCV(MultinomialNB(), {'alpha': alpha_params}, cv = 5, return_train_score=False)
            clf.fit(x, y_data)

            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_parameters["best_alpha"] = clf.best_params_["alpha"]
                best_parameters["best_ngram_range"] = ngram_range_params[i]



        st.dataframe(pd.DataFrame(best_parameters))

        st.markdown("***GET JUPYTER NOTEBOOK HERE FROM GITHUB***  ")
        st.markdown("[jupyter notebook](https://raw.githubusercontent.com/CyrilleMesue/dna-seq-classifier/main/DNA%20Sequencing%20and%20applying%20Classifier.ipynb)")

#        st.markdown("### Save Models")
        # vec name
#        v_filepath = 'tmp/dna_vectorizer.sav'
#        pickle.dump(cv, open(v_filepath, 'wb'))

#        st.download_button("Download Vectorizer Model", open(v_filepath, 'rb'), file_name="dna_vectorizer.sav")



















