# Sentiment Analysis of Movie Reviews

This repository contains code for performing sentiment analysis on movie reviews using various machine learning techniques. The goal is to classify movie reviews as either positive or negative based on their sentiment.

## Dataset
- The dataset used for training and testing the sentiment analysis models is the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative.

## Code Overview
- `sentiment_analysis.ipynb`: Jupyter Notebook containing the code for sentiment analysis using different machine learning models.
- `README.md`: This file, providing an overview of the project and instructions for running the code.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, re, nltk, sklearn, gensim

## Instructions
1. Clone the repository to your local machine.
2. Open `sentiment_analysis.ipynb` in Jupyter Notebook.
3. Ensure that all necessary libraries are installed.
4. Run the code cells sequentially to train and test the sentiment analysis models.
5. View the classification reports and evaluation metrics to assess the performance of each model.

## Models Implemented
1. **Bag of Words with Random Forest Classifier**: Uses CountVectorizer to convert text data into numerical features and trains a Random Forest Classifier for sentiment analysis.
2. **TF-IDF with Random Forest Classifier**: Utilizes TF-IDF Vectorization to represent text data and trains a Random Forest Classifier for sentiment analysis.
3. **Word2Vec with Random Forest Classifier**: Implements Word2Vec embedding to convert text data into word vectors and trains a Random Forest Classifier for sentiment analysis.

## Evaluation
- The performance of each model is evaluated using classification reports, which include precision, recall, F1-score, and accuracy metrics.
- Additionally, the Area Under the Curve (AUC) for Receiver Operating Characteristic (ROC) curves is calculated to assess model performance.

## Conclusion
This project demonstrates the effectiveness of different machine learning techniques for sentiment analysis of movie reviews. By training and testing multiple models, we can identify the most suitable approach for accurately classifying movie sentiments.

For any questions or feedback, please feel free to contact the repository owner.
