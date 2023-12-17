Authorship Attribution for Neural Text Generation- **RUN APP**(https://apporfinder-grtlgb4yxauid3dye9l9wd.streamlit.app/)

#Introduction 

With the developments in the field of chatbots which could mimic the human language, there’s a high ground for the models to be used as part of the deep fakes. Thus in this project we have analyzed various prior work in order to solve the following three questions.

Same or not: Whether the generated text is from the same NLG method (human) or not.
Human vs bot: Whether the text is from a human or an NLG method.
Which NLG method: Whether the text is from the ith NLG method from a set of k methods.

The above questions are typical binary or multi label classification problems. Hence the papers reviewed would deal with the domain of the classification of text using different Machine Learning, Deep Learning models such as the RNN and so on.

#Dataset

In our study, we investigated three primary tasks using datasets from ten Natural Language Generation (NLG) methods: CTRL, GPT, GPT2, GPT3, Instruct GPT, GROVER, XLM, XLNET, PPLM, FAIR, and also human-generated content. Our objectives were:

Determining Correspondence: Assessing if two pieces of text are correlated.
Classifying Text Types: Differentiating between two distinct classes of text.
Identifying NLG Methods: Recognizing the specific NLG method behind a given text.
To evaluate our models, we prepared both balanced (1:1 ratio) and imbalanced (1:10 ratio) datasets. Moreover, we conducted a comprehensive Reddit case study involving GPT3, InstructGPT, and human contributions across various topics. Here, we used balanced (1:1 ratio) and imbalanced (1:2 ratio) datasets for all three tasks.

Our research provides crucial insights into the effectiveness of various text analysis and classification methods. We explored five distinct methodologies and their corresponding models, each carefully chosen and tailored to the specific requirements of the tasks. This approach allowed us to thoroughly evaluate the performance of both machine learning classifiers and deep learning architectures under diverse conditions.

#Methods Used

In our research, we employed a range of methodologies to approach the task of authorship attribution and text classification, using datasets from various NLG methods and human-generated content. The methods we used were as follows:

First Method: This approach focused on basic preprocessing steps, including removing punctuations, eliminating stopwords, and converting all text to lowercase. Tokenization was done using a standard approach. The core model was a Bidirectional Long Short-Term Memory (BiLSTM) network with two layers, comprising 64 units in the first layer and 32 units in the second.

Second Method: This method differed from the first by foregoing any preprocessing and instead employed the BERT tokenizer for tokenization. The BiLSTM architecture was consistent with the first method, maintaining the same configuration of 64 units in the first layer and 32 in the second.

Third Method: Similar to the second method, we opted out of preprocessing and utilized the BERT tokenizer. Additionally, this method involved extracting stylometric features such as average word length, average sentence length, vocabulary size, lexical diversity, noun count, verb count, and adjective count. These features were then integrated into the BiLSTM model used in the previous methods.

Fourth Method: This method combined stylometric analysis with the BERT tokenizer and a modified BiLSTM architecture. It included the extraction of various stylometric features and the use of global max-pooling, dropout, and batch normalization layers in the BiLSTM model. These additions aimed to improve model generalization and stability in learning.

Fifth Method: The fifth approach was distinct, involving a grid search to define a range of hyperparameters for an XGBoost classifier model. This method also utilized the Synthetic Minority Over-sampling Technique (SMOTE) to address data imbalance.

#Flow Chat:- 

![image](https://github.com/HarinathCingapuram94/AuthorFinder/assets/60059816/67f285de-f449-44b5-8908-1a0ad88d246b)

#Results:-

Task 1:- 
![image](https://github.com/HarinathCingapuram94/AuthorFinder/assets/60059816/7f927a57-0df3-4d84-bebc-29973744c2af)

Task2:- 
![image](https://github.com/HarinathCingapuram94/AuthorFinder/assets/60059816/e43e2eee-3f3c-4fdf-87f0-deedec26504b)

Task3:- 
![image](https://github.com/HarinathCingapuram94/AuthorFinder/assets/60059816/01a5eeaf-3efd-42c2-a748-edef4923099e)



#Conclusion:-

Our study's implemented models exhibited noteworthy performance across a variety of tasks, displaying particular strengths in certain areas. Notably, in some cases, models developed by our team and the Random Forest model surpassed others in effectiveness, highlighting their robustness and efficiency in handling complex classification tasks.

A key discovery from our research was the effectiveness of stylometric features in extracting meaningful information from text data. These features, when combined with Part-of-Speech (POS) estimation, provided deep insights into the grammatical and syntactic structures of the chat data. This combination proved particularly valuable in discerning subtle nuances that differentiate human-generated texts from those produced by NLG methods.

Furthermore, the incorporation of a robust pre-trained tokenizer, such as BERT, significantly enhanced our models' performance. This underscores the potential benefits of integrating traditional stylometric analysis with advanced deep learning techniques in authorship attribution tasks. The synergy between these approaches contributed significantly to the accuracy and reliability of our models.

These findings are not just a testament to the success of our current research but also pave the way for future exploration in this field. They suggest potential avenues for further improvements and novel approaches that can be pursued in future studies. Overall, our research contributes valuable insights to the domain of authorship attribution, underscoring the importance of combining diverse methodologies to tackle the challenges presented by NLG technologies.

#Features

Model Selection: Choose from different models including XGBoost and BiLSTM to perform the classification.
NLG Method Detection: Capable of distinguishing texts generated by various NLG methods like GPT-2, GPT-3, etc.
User-Friendly Interface: Simple and intuitive UI for easy interaction with the application.
How It Works
The application uses a combination of TensorFlow models and NLP techniques to analyze the input text. It preprocesses the data using NLTK, vectorizes it, and then feeds it into the selected machine learning model for classification.

#Technologies Used

Streamlit for web app development
TensorFlow and Keras for model building and predictions
NLTK for natural language processing
Scikit-learn for additional machine learning utilities
Setup and Installation

For an in-depth understanding of the methodologies and technologies used in this project, refer to our [Research Paper](https://drive.google.com/file/d/1STTkT4chq314ALw1R3VVPjgmVmxFWBjZ/view?usp=sharing).
A detailed overview of the project can be found in our [Presentation](https://drive.google.com/file/d/1r7W4T2jD2-hCr2d0ZY09Wkoh70M-6BKl/view?usp=sharing).
