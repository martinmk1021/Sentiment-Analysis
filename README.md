# Project: Sentiment Analysis on Amazon Review Polarity Dataset

## Overview
This project performs sentiment analysis on the Amazon Review Polarity dataset. The dataset contains product reviews from Amazon, where reviews with scores 1 and 2 are labeled as negative (class 1), and those with scores 4 and 5 are labeled as positive (class 2). The dataset contains 1,800,000 training samples and 200,000 testing samples for each class (negative and positive).

## Objective
The objective of this project is to classify Amazon reviews into positive and negative sentiments using a machine learning model. The sentiment labels are as follows:
- **Class 1**: Negative sentiment (1-2 rating)
- **Class 2**: Positive sentiment (4-5 rating)

## Dataset Details
The dataset used is from Kaggle and can be found [here](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews). It consists of:
- **Polarity**: 1 for negative and 2 for positive.
- **Title**: Review heading.
- **Text**: Review body.

We used 50% of each class from the training data randomly for model training, and the complete test set was used for evaluation.

## Methodology
1. **Data Preprocessing**:
   - Cleaned the review text by removing special characters and non-alphabetic characters.
   - Utilized **Logistic Regression** for sentiment classification.
   
2. **Model Evaluation**:
   - Cross-validation was performed to assess the model's robustness, and the following metrics were obtained:
     - **Cross-Validation Accuracy**: 90.16%
     - **Test Accuracy**: 90.20%
     - **Precision, Recall, F1-Score** (for both classes):
       - **Precision**: 0.90
       - **Recall**: 0.90
       - **F1-Score**: 0.90

3. **Final Results**:
   - **Cross-Validation Accuracy Scores**: 
     ```
     [0.90207222, 0.90098056, 0.90078889, 0.90263889, 0.90172778]
     ```
   - **Mean CV Accuracy**: 0.9016
   - **Standard Deviation of CV Accuracy**: 0.0007

   - **Test Accuracy**: 0.9020

   - **Classification Report**:
     ```
                precision    recall  f1-score   support

            1      0.90       0.90      0.90    200000
            2      0.90       0.90      0.90    200000

     accuracy                           0.90    400000
     macro avg     0.90      0.90       0.90    400000
     weighted avg  0.90      0.90       0.90    400000
     ```

## Tools & Technologies
- **Programming Language**: Python
- **Libraries Used**: scikit-learn, numpy, pandas, matplotlib, seaborn
- **Model**: Logistic Regression

## Future Work
- Exploring other models like SVM, XGBoost, and ensemble methods to potentially improve the accuracy.
- Experimenting with different feature extraction techniques, such as TF-IDF or word embeddings, to further enhance model performance.

## Citation
The dataset is from the [Kaggle Amazon Reviews dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews), and for more information, refer to the original Stanford Network Analysis Project (SNAP) work available in ["Character-level Convolutional Networks for Text Classification"](https://papers.nips.cc/paper/2015/hash/9f40a69f0a1de01f4332851d32b58c94-Abstract.html) (NIPS 2015).
