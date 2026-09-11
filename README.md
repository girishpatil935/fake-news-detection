# Fake News Detection using Machine Learning

> **GitHub Setup — Replace These Placeholders**
>
> - **GitHub Username:** `girishpatil935`
> - **Repository Name:** `https://github.com/girishpatil935/fake-news-detection`
> - **GitHub URL:** `https://github.com/girishpatil935`



A machine learning project for detecting whether a news statement is
**Fake** or **Real** using traditional NLP and classification
techniques.

The project uses **TF-IDF text features** and multiple machine learning
classifiers, followed by evaluation on an external **IFND dataset** to
measure model generalization.

------------------------------------------------------------------------

## Project Overview

The pipeline consists of:

1.  Data understanding and preprocessing
2.  Text cleaning
3.  TF-IDF feature extraction
4.  Machine learning model training
5.  Model evaluation
6.  External evaluation using IFND
7.  Model comparison and optimization

### Label Mapping

  Label   Meaning
  ------- ---------
  `0`     Fake
  `1`     Real

------------------------------------------------------------------------

## Dataset

The project works with news datasets containing:

-   `Statement` --- news text/content
-   `Label` --- Fake or Real classification

### External IFND Dataset

The IFND dataset is used as an **external evaluation dataset**.

The evaluated dataset contains:

-   **56,714 statements**
-   **37,800 Real/True**
-   **18,914 Fake**
-   No missing values in the evaluated `Statement` and `Label` columns

------------------------------------------------------------------------

## Machine Learning Pipeline

``` text
Raw News Data
      |
      v
Data Cleaning & Preprocessing
      |
      v
Text Data
      |
      v
TF-IDF Vectorization
      |
      v
Machine Learning Model
      |
      +--> Logistic Regression
      +--> Linear SVM
      +--> Multinomial Naive Bayes
      +--> Random Forest
      |
      v
Predictions
      |
      v
Evaluation
      |
      +--> Accuracy
      +--> Precision
      +--> Recall
      +--> F1 Score
      +--> Confusion Matrix
```

------------------------------------------------------------------------

## Models Tested

### Logistic Regression

Initial external IFND evaluation:

  Metric             Score
  ----------- ------------
  Accuracy      **64.58%**
  Precision     **66.08%**
  Recall        **96.28%**
  F1 Score      **78.37%**

The model shows a strong tendency toward predicting the Real class.
Further work focuses on threshold tuning, hyperparameter tuning, TF-IDF
optimization, and class-imbalance handling.

### Linear SVM

Initial external IFND evaluation:

  Metric             Score
  ----------- ------------
  Accuracy      **63.06%**
  Precision     **65.77%**
  Recall        **92.96%**
  F1 Score      **77.03%**

### Multinomial Naive Bayes

Initial external IFND evaluation:

  Metric             Score
  ----------- ------------
  Accuracy      **26.94%**
  Precision     **41.67%**
  Recall        **24.07%**
  F1 Score      **30.52%**

Naive Bayes currently performs poorly on the external dataset.

### Random Forest

Initial external IFND evaluation:

  Metric             Score
  ----------- ------------
  Accuracy      **66.57%**
  Precision     **66.63%**
  Recall        **99.88%**
  F1 Score      **79.93%**

Although Random Forest has the highest raw accuracy, its Fake-class
performance is extremely poor. It predicts almost all external samples
as Real, making raw accuracy misleading.

------------------------------------------------------------------------

## Why Accuracy Alone Is Not Enough

The IFND dataset contains more Real samples than Fake samples.

A model can therefore obtain relatively high accuracy by predicting most
samples as Real while failing to detect Fake news.

For this reason, this project evaluates:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix
-   Class-wise performance

The objective is to build a model that can meaningfully distinguish
**Fake** from **Real**, not simply maximize accuracy.

------------------------------------------------------------------------

## Current Model Direction

**Logistic Regression** is currently selected for further optimization.

Planned improvements:

### 1. Decision Threshold Tuning

Instead of relying only on the default `0.5` decision threshold,
different thresholds will be tested to find a better balance between
Fake and Real predictions.

### 2. Hyperparameter Tuning

The following parameters will be explored:

-   `C`
-   `class_weight`
-   solver configuration

### 3. TF-IDF Optimization

Experiments will include:

-   `ngram_range`
-   `min_df`
-   `max_df`
-   `sublinear_tf`
-   vocabulary size
-   word n-grams
-   character n-grams

### 4. Class Imbalance Handling

Possible approaches include:

-   `class_weight="balanced"`
-   resampling
-   decision-threshold optimization

### 5. Error Analysis

Misclassified Fake and Real statements will be inspected to understand
the model's failure patterns.

------------------------------------------------------------------------

## Project Structure

``` text
fake-news-detection/
|
+-- data/
|   +-- raw/
|   |   +-- Fake.csv
|   |   +-- True.csv
|   |   +-- IFND.csv
|   |   +-- fake_clean.csv
|   |   +-- real_clean.csv
|   |
|   +-- processed/
|
+-- models/
|
+-- notebooks/
|   +-- 01_data_understanding.ipynb
|   +-- 02_new_dataset.ipynb
|
+-- src/
|
+-- requirements.txt
|
+-- README.md
```

> Large datasets and trained model files may be excluded from GitHub
> depending on their size and dataset licensing.

------------------------------------------------------------------------

## Technologies Used

### Programming

-   Python

### Data Processing

-   Pandas
-   NumPy

### NLP

-   Scikit-learn
-   TF-IDF Vectorization

### Machine Learning

-   Logistic Regression
-   Linear Support Vector Machine
-   Multinomial Naive Bayes
-   Random Forest

### Evaluation

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix
-   Classification Report

### Development

-   Jupyter Notebook
-   Visual Studio Code
-   Git
-   GitHub

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-GITHUB-REPOSITORY>.git  # REPLACE BOTH PLACEHOLDERS
cd <YOUR-GITHUB-REPOSITORY>  # REPLACE <YOUR-GITHUB-REPOSITORY>
```

Create a virtual environment:

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

macOS/Linux:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Project

Start with:

``` text
notebooks/01_data_understanding.ipynb
```

to inspect and preprocess the data.

Then run:

``` text
notebooks/02_new_dataset.ipynb
```

to perform external IFND evaluation and compare the trained models.

------------------------------------------------------------------------

## Current Results

  Model                   Accuracy   Precision   Recall       F1
  --------------------- ---------- ----------- -------- --------
  Logistic Regression       64.58%      66.08%   96.28%   78.37%
  Linear SVM                63.06%      65.77%   92.96%   77.03%
  Naive Bayes               26.94%      41.67%   24.07%   30.52%
  Random Forest             66.57%      66.63%   99.88%   79.93%

### Important Observation

Random Forest has the highest raw accuracy, but it identifies almost
none of the Fake samples in the external dataset.

Therefore, the project is currently focusing on improving **Logistic
Regression** rather than selecting a model based only on accuracy.

------------------------------------------------------------------------

## Future Work

-   [ ] Tune Logistic Regression hyperparameters
-   [ ] Optimize classification threshold
-   [ ] Optimize TF-IDF configuration
-   [ ] Handle class imbalance
-   [ ] Perform cross-validation
-   [ ] Conduct detailed error analysis
-   [ ] Compare word and character n-grams
-   [ ] Test ensemble approaches
-   [ ] Evaluate additional external datasets
-   [ ] Build a real-time prediction interface
-   [ ] Deploy the final model as an API

------------------------------------------------------------------------

## Disclaimer

This project is an academic and machine-learning experiment.

A classifier can learn statistical patterns in text, but it cannot
independently verify whether a real-world claim is factually true.
Predictions should therefore be treated as model outputs rather than
verified facts.

------------------------------------------------------------------------

## GitHub Repository

Replace the placeholders below before publishing:

```text
https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-GITHUB-REPOSITORY>
```

- **Username:** `girishpatil935`
- **Repository:** `https://github.com/girishpatil935/fake-news-detection`

## Author

**Girish Patil**

GitHub: `https://github.com/girishpatil935`  

------------------------------------------------------------------------

## License
**Note : The project is still under training**

This project is intended for educational and research purposes. Add an
appropriate open-source license after checking the licenses of the
datasets used.
