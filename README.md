# Project Description
__Problem__: The aim of the project is to categorize the toxic comments based on the types of toxicity. Examples of toxicity types can be toxic, severely toxic, obscene, threat, insult, identity hate. Different machine learning techniques like Logistic Regression, Support Vector Machines and Naive Bayes are implemented to determine the 6 types of toxic comments.
__Data__: The dataset we are using for toxic comment classification is taken from Kaggle competition which can be found at Kaggle. Dataset has a large number of comments from Wikipedia talk page edits. They have been labeled by human raters for toxic behavior.

|         | toxic | severe_toxic | obscene | threat | insult | identity-hate |
|:-------:|:-----:|:------------:|:-------:|:------:|:------:|:-------------:|
|Count    |15294|1595|8449|478|7877|1405|
|Percentage|9.5%|0.9%|5.2%|0.2%|4.9%|0.8%|
![alt text][https://github.com/mrthlinh/toxic-comment-classification/blob/master/pic/distribution.png]

Data are highly imbalanced

__Execute project__:

```scala
spark-submit --class <Class Name> project_2.11-0.1.jar <name of label> <output directory>

*<Class Name>       : We have used four models thus you can execute any one of them.
					"TrainingNB", "TrainingLR", "TrainingNN" and "TrainingSVM"


*<name of label >   : For which label you want to run the jar e.g. "toxic", "threat"
*<output directory> : Directory to store your result.

Sample execution
>spark-submit --class TrainingNB project_2.11-0.1.jar toxic output
```
After executing there will be a new output folder in this directory with desired results.

# Proposed Solution and Methods
Data Analysis shows that all the labels are closely related and some labels have distinctly high correlation compared to others, and we are trying to predict the probability for all the labels in parallel. Training and fitting the model for each label will give us better results.

EDA shows that the number of comments which were actually tagged were relatively low compared to the total number of comments in the dataset. We have used pipelines for tokenizing, removing stop-words, feature-extraction, hyper-parameter tuning and cross-validation.

To handle imbalanced data we did 3 experiments as follows:
1. Define higher weight for label ‘1’:
Depending on how imbalanced the data is, we put higher weight on the minority label to
force the classifier to work harder on label ‘1’.
For example, if the ratio of label ‘1’ to the label ‘0’ is 1:9 we will put 0.9 as the weight factor on label ‘1’.
2. Over-sampling label ‘1’ (minority class):
To reduce the imbalance, we simply duplicate samples with label ‘1’ to achieve 1:1 as the ratio of negative samples to positive samples. But there is a downfall for this method. It might result in over-fitting.
3. Under-sampling label ‘0’ (majority class):
To reduce the imbalance, this time we reduce the number of negative samples with label ‘0’ to get the ratio of negative samples to positive samples closer to 1:1. This also has a disadvantage of losing potentially important data.

For each experiment, we perform 10-fold cross validations to pick the best parameters and evaluation on the test data. Results are discussed further in Result section

# Results
For this problem while under-sampling does not work well, higher-weight and over-sampling give us better results.

|  Label  | Model | F1-score for class "1" |Method|
|:-------:|:-----:|:----------------------:|:----:|
|Toxic| Neural Network |0.69| higher weight|
|Severe Toxic| Naïve Bayes| 0.64|over sampling|
|Obscene| Neural Network | 0.70|higher weight|
|Threat| Neural Network| 0.23|higher weight|
|Insult| Neural Network| 0.59|higher weight|
|Identity Hate| Neural Network| 0.28|higher weight|

# Conclusion
After considering and comparing the models based on precision, recall, accuracy and F1-
score, we conclude that the Neural Network model trained on clean data has the best
results with one exception for Naïve Bayes of label “severe toxic”. However, consider the
number of features we used, while other models need more than 200,000 features, Neural
Network with only 100 features from Word2Vec can give better results. Furthermore, as
we increase the number of hidden layer and iteration number, the result get better.

# References
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
https://spark.apache.org/mllib/
https://spark.apache.org/docs/latest/mllib-naive-bayes.html
https://spark.apache.org/docs/latest/ml-classification-regression.html#logisticregression
https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-supportvector-machines-svms
https://plot.ly
