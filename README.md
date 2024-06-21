# Fake News Detection Project

## Overview
This project focuses on addressing the critical issue of fake news detection using advanced data science and machine learning techniques. The solution leverages undersampling methods and fine-tuned LSTM models to achieve high accuracy in identifying fake news.

## Key Features
### Undersampling Methods
- **Implementation**: Applied NearMiss undersampling technique on the Kaggle Fake and Real News Dataset.
- **Impact**: Improved model accuracy by 10%, significantly enhancing the detection of misinformation.

### Fake News Classifier
- **Model**: Developed a robust fake news classifier using LSTM (Long Short-Term Memory) models.
- **Feature Extraction**: Utilized the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer for converting text data into numerical features.
- **Performance**: Achieved an impressive 98% accuracy in detecting fake news.

## Dataset
- **Source**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Content**: The dataset consists of 44,898 news articles in English, labeled as "FAKE" or "REAL", covering various categories such as politics, sports, entertainment, and more.
- **Columns**: Title, Text (body text), Subject, and Date (publish date).

## Approach
### Data Preprocessing
- **Text Cleaning**: Removed stopwords, punctuation, and special characters.
- **Feature Extraction**: Employed TF-IDF vectorizer to convert textual data into numerical features.

### Addressing Data Imbalance
- **Technique**: Applied NearMiss undersampling to balance the dataset.
- **Effectiveness**: Evaluated the impact of undersampling by comparing model performance on balanced and imbalanced datasets.

### Model Selection and Training
- **Basic Models**: Naive Bayes, Logistic Regression, Support Vector Machine (SVM).
- **Advanced Models**: Decision Trees, Random Forest, LSTM networks.
- **Hyperparameter Tuning**: Used grid search and random search methods to find optimal hyperparameters.

### Evaluation Metrics
- **Metrics Used**: Accuracy, Precision, Recall, and F1 Score.
- **Emphasis**: Placed particular emphasis on Recall to ensure fake news is correctly identified.

## Results
- **Accuracy**: Achieved a 98% accuracy with the fine-tuned LSTM model.
- **Improvement**: Observed a 10% increase in accuracy with the implementation of NearMiss undersampling.

## Conclusion
This project demonstrates the effectiveness of using LSTM models and undersampling techniques in detecting fake news. By improving model accuracy and addressing data imbalance, the solution provides a robust tool for combating misinformation.

## Future Work
- **Explore Different Architectures**: Investigate the impact of other machine learning and deep learning models.
- **Multimodal Approach**: Incorporate image and video analysis alongside text analysis.
- **Transfer Learning**: Leverage pre-trained models to improve performance and reduce training times.
- **Explainability**: Integrate techniques like LIME (Local Interpretable Model-Agnostic Explanations) to understand model decision-making processes.
- **Real-time Application**: Develop real-time detection systems for social media platforms and news websites.

## Installation and Usage
### Prerequisites
- Python
- TensorFlow
- NLTK
- Pandas
- Scikit-learn

