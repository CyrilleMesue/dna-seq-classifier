#  DNA Sequence Classification 


### About App 

This app is a deployment of a DNA classification mini project using machine learning. A multinomial naive Bayes classifier is trained on the dataset and used to classifiy coding DNA sequences. 


The dataset for this study was gotten from kaggle and can be accessed through the link below:  
https://www.kaggle.com/nageshsingh/dna-sequence-dataset

And this code was adapted from : https://github.com/krishnaik06/DNA-Sequencing-Classifier

The model have been trained and validated using a 10-fold cross validation and achieved an average accuracy of 98.2 on the human dataset. These models (countvectoriizer and multinomial naive Bayes classifier) can be used to make predictions on new datasets. All that is needed is to enter a DNA text string or upload a text file in the **Predict DNA class** on the taskbar.  


You can also train on your own dataset provided the data is in text format where the DNA(label = sequence) is seprated from the labels(label = class) by a comma. You can play with the **Machine Learning Pipeline** on the taskbar for this. It might take a few minutes to completely run the code in the background. 

### How to predict DNA class

 

### Train Models on New Datasets 




## Technology Stack 

1. Python 
2. Streamlit 
3. Pandas
4. Scikit-Learn
5. markdown


## How to Run 

- Clone the repository
- Setup Virtual environment
```
$ python3 -m venv env
```
- Activate the virtual environment
```
$ source env/bin/activate
```
- Install dependencies using
```
$ pip install -r requirements.txt
```
- Run Streamlit
```
$ streamlit run app.py
```


## Contributors 

<table>
  <tr>
    <td align="center"><a href="https://github.com/prakharrathi25"><img src="https://avatars.githubusercontent.com/u/38958532?v=4" width="100px;" alt=""/><br /><sub><b>Cyrille M. NJUME</b></sub></a><br /></td>
  </tr>
</table>

## References 

[1] krishnaik06: [https://github.com/krishnaik06/DNA-Sequencing-Classifier](https://github.com/krishnaik06/DNA-Sequencing-Classifier)

[2] Prakhar Rathi: [Creating Multipage applications using Streamlit (efficiently!)](https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030)
[3]  Nagesh Singh Chauhan : [Dataset](https://www.kaggle.com/nageshsingh/dna-sequence-dataset)

## Contact

For any feedback or queries, please reach out to [cyrillemesue@gmail.com](cyrillemesue@gmail.com).
