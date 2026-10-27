# Super-strong Machine Learning Identification Model Based on Feature Extraction of Fault Signals

\n        \n  
New addition two: Bearing Fault Characteristics - SHAP Model 3D Visualization
The dataset includes a complete set of bearing faults, as well as the generated fault signal dataset, classification data, and label set. The corresponding codes can be run, along with feature extraction code.
The accuracy rate is 100%, with XGBoost's classification model performing best, achieving the highest performance on both training and testing sets, with the highest model score. It has achieved good results in bearing fault diagnosis. By accurately capturing the features and patterns of signals, it can accurately identify and classify different types and degrees of bearing faults, achieving a 100% accuracy rate. The model is fast, performs well, and has high innovation. To date, it is the most accurate method among previous articles. Its innovation is also significant!
This includes data preprocessing code, data feature extraction, and XGBoost classification code.
Environment: Python 3.9
SHAP model visualization + parameter search strategy in bearing fault diagnosis application
Preface
This paper is based on the bearing data from Case Western Reserve University (CWRU) and employs feature extraction and machine learning methods to identify faults. Subsequently, we introduce a parameter search strategy based on the XGBoos model, and analyze the results using SHAP model visualization techniques. For a detailed introduction to the bearing data from CWRU, please refer to the following article:  
Python-Case Western Reserve University (CWRU) Bearing Data Interpretation and Classification Processing  
Visualization Results:  
Ten-class confusion matrix
\n        \n  
Global feature importance map
\n        \n  
10. Category ROC curve and AUC values:
\n        \n  
Comparison of true values and predicted values for tags
\n        \n  
1. Dataset and feature extraction
\t1.1 Dataset import
\tRefer to the previous article for preprocessing of fault classification, \u0026 Caissy State University bearing data \u0026 10-classification dataset:
\n        \n  
1. Data Reading Form and Preprocessing Ideas
1.2 Feature Extraction of Fault Signals
1.3 Selection of 13 indicators such as kurtosis, entropy value, fractal value, waveform index, spectral index, frequency domain index, statistical feature, vibration feature to capture multi-scale features of bearing signals for training and recognition of machine learning models.
2 XGBoost Fault Diagnosis Model Based on Parameter Search Strategy
2.1 Parameter Search Strategy: Large-step rough search followed by small-step fine search
2.2 For the important parameters of the XGBoost model: number of trees, depth of the tree, learning rate, a large-range rough search is adopted initially to lock in approximate parameter range values, and then a small-step fine search is used to accurately locate the parameter values.
\n        \n  
\t2.2 Broad-Band Approximate Search  
\tGrid search is adopted to confirm the approximate parameter range values.
\n        \n  
\t2.2 Small Steps for Extensive Search  
\tStep 1, search the number of trees in the search tree
\n        \n  
Step 2: Searching the depth of the tree
\n        \n  
Step 3: Search for learning rate.
\n        \n  
3. XGBoost Model Evaluation and Visualization
\t3.1 Model Scores, Accuracy, Precision, Recall, F1 Score
\n        \n  
\t3.2 Failure classification confusion matrix
\n        \n  
3.3 SHAP Model Visualization
\n        \n  
SHAP (SHapley Additive exPlanations) is a method for explaining the predictions of machine learning models. It measures the contribution of each feature to the model output. By calculating SHAP values, we can understand the impact of each feature on the model's prediction, thereby better comprehending the classification prediction process of the model. This interpretability is very helpful in enhancing the credibility and explainability of the model. It allows further analysis of the contribution of features extracted from fault signals to bearing fault diagnosis.
The average (SHAP value) (average impact on the size of the model output), can clearly show that 13 features contribute to the classification of each category!
\n        \n  
High-innovation!!! Students needing to publish papers can use this immediately.
Preface:
Bearing fault diagnosis faces some challenges. Firstly, bearing fault signals are usually complex nonlinear signals, containing rich frequency and time domain information. Traditional analysis methods often struggle to fully capture the features in the fault signal. Secondly, bearing fault signals often accompany background noise and interference signals, increasing the difficulty of extracting and identifying the fault signals. Finally, there are various types and degrees of bearing faults, requiring a flexible method to distinguish different fault patterns.
This paper is based on CWRU's bearing data and uses feature extraction and machine learning methods for fault identification. Feature extraction involves extracting meaningful features from the raw signal to reduce data dimensionality and capture key information about the signal. Machine learning methods can establish models based on these features and identify different fault patterns through training and learning. This combination enhances the accuracy and efficiency of bearing fault diagnosis. A detailed introduction to CWRU's bearing data can be found in the following article:
Python-CWRU's Bearing Data Interpretation and Classification Processing
1. Dataset and Feature Extraction
1.1 Dataset Import
Refer to previous articles for preprocessing of fault classification with 10 categories, using CWRU's bearing data set:
Data reading form and preprocessing ideas
Created dataset with signal time step size 1024 and overlap rate 0.5
Formed 2330 samples, each sample length 1024, plus one label category.
1.2 Fault Signal Feature Extraction
(1) Kurtosis (Kurtosis): Measuring the sharpness of the signal, used to detect high-frequency components in the signal
(2) Entropy (Entropy): Measuring the complexity and randomness of the signal, used to detect the spectral characteristics of the signal
(3) Fractal Dimension: Measuring the self-similarity and complexity of signals, used for analyzing the fractal characteristics of signals.
(4) Waveform Indicators: Including peak factor, pulse factor, margin factor, etc., used for analyzing the time-domain characteristics of signals.
(5) Spectral Indicators: Includes peak frequency, energy ratio, spectral line indicators, etc., used for analyzing the frequency domain characteristics of signals.
(6) Frequency Domain Indicators (Time-Frequency Indicators): Including instantaneous frequency, instantaneous energy, etc., used for analyzing the time-frequency characteristics of signals.
(7) Statistical Features (Statistical Features): Including mean, variance, skewness, etc., used to describe the statistical characteristics of signals.
(8) Wavelet Packet Features: Features extracted through wavelet transformation, used for analyzing the time-frequency local characteristics of signals.
(9) Vibration Characteristics (Vibration Features): Including peak vibration and effective value vibration, etc., used to describe the vibration characteristics of signals.
Various feature extraction methods were selected to capture different characteristics of the signal, resulting in a total of 13 features indicators across 9 categories for training and recognition of machine learning models.
2. Superior Model XGBoost - Principles Introduction
2.1 Principle Introduction
\tPaper link: \t
\tXGBoost | Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining  
GBoost model (Extreme Gradient Boosting) is a gradient boosting framework developed by Tianqi Chen in 2014 and widely used in the field of machine learning. The core idea behind XGBoost is to iteratively train multiple weak learners and combine them to achieve powerful predictive capabilities. It improves and optimizes the gradient boosting algorithm, characterized by its efficiency, flexibility, and scalability.

Here are some key features and principles of XGBoost:

1. Gradient Boosting: XGBoost employs the gradient boosting algorithm, also known as the Boosting algorithm. It iteratively trains multiple weak learners and optimizes the model's predictive ability through gradient descent. Each weak learner is trained on the residuals of the previous weak learner, gradually reducing prediction errors.

2. Tree-based Model: XGBoost adopts a tree-based model, namely decision trees. Decision trees are very flexible and interpretable models that can learn complex nonlinear relationships. XGBoost uses CART (Classification and Regression Trees) as the default base learner, where each decision tree is implemented by continuously splitting the feature space for classification or regression tasks.

3. Regularization Strategy: To prevent overfitting, XGBoost introduces regularization strategies. It limits the model's learning capability by controlling the complexity of the decision trees. Common regularization strategies include limiting the maximum depth of the decision trees, setting the minimum number of samples per leaf node, and weight decay for leaf nodes.
4. Feature selection and splitting: XGBoost maximizes the model's gain during the construction of decision trees by feature selection and splitting. Feature selection is based on some evaluation criterion (such as information gain or Gini coefficient), selecting the most advantageous features for current nodes. Feature splitting is the process of determining the points where features are divided, ensuring that the sub-nodes after division can minimize prediction errors to the greatest extent.
5. Parallel Computing: To enhance the training speed of the model, XGBoost employs a parallel computing strategy. It decomposes the training task into multiple subtasks through techniques such as multi-threading and distributed computing, and performs computation simultaneously on different processors. This approach can significantly accelerate the model's training process, particularly when dealing with large-scale datasets.
6. Custom Loss Functions: XGBoost allows users to define custom loss functions to suit different tasks and requirements. Users can define loss functions that are suitable for their specific problem characteristics and use them during the model training process.
The XGBoost model, leveraging the gradient boosting algorithm and tree-based models, has achieved excellent results in numerous machine learning tasks, including classification, regression, ranking, and recommendation. By utilizing its efficient, flexible, and scalable features, we employ XGBoost to construct a gradient boosting model for bearing fault identification through iteratively training multiple decision trees.
2.2 Feature dataset creation
3. Model Evaluation and Comparison
3.1 Random Forest Classification Model
Model score, accuracy, precision, recall, F1 Score
Confusion Matrix
\t  
\t3.2 Support Vector Machine (SVM) Classification Model
\tModel scores, accuracy, precision, recall, F1 Score
\tFault classification confusion matrix:
\t3.3 XGBoost Classification Model
\tModel scores, accuracy, precision, recall, F1 Score
\tFault classification confusion matrix:
\t  
Experimental results show that the various features extracted have a certain contribution to bearing fault diagnosis. Skewness, entropy values, and fractal values can help capture the sharpness, complexity, and self-similarity of signals, thereby effectively distinguishing different types of faults. Waveform indicators, spectral indicators, and frequency domain indicators can provide time-domain and frequency-domain characteristics of signals, aiding in the recognition of time-frequency characteristics of faults. Statistical features, wavelet packet features, and vibration features can describe the statistical and vibrational characteristics of signals, thus better distinguishing fault patterns.
Comparative analysis reveals that the XGBoost classification model performs best, with the highest model score on both training and testing sets, achieving good results in bearing fault diagnosis. By accurately capturing the features and patterns of signals, it can accurately identify and classify different types and degrees of bearing faults, achieving a 100% accuracy rate, fast speed, high performance, and high innovation.
Click to download: Complete original data, Python code

## Images

![Super-strong machine learning recognition model based on fault signal feature extraction](01.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](02.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](03.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](04.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](05.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](06.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](07.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](08.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](10.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](11.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](12.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](13.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](14.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](15.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](16.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](17.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](18.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](19.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](20.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](21.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](22.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](23.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](24.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](25.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](26.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](27.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](28.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](29.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](30.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](31.png)
![Super-strong machine learning recognition model based on fault signal feature extraction](32.png)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
