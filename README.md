# Basic Machine Learning Algorithms using numpy
All algorithms described below are implemented in `utils.py`. Each algorithm is accompanied by examples below its class definition, allowing for quick implementation and testing.
## Supervised Learning
### LogisticRegression
Optimizer is Newton's Method
### Locally Weighted Linear Regression
Add the attention mechanism into linear regression to fit a non-linear model
### Gaussian Descriminant Analysis
The supervised learning version of Gaussian Mixture Model
### Naive Bayes
Used for language processing, eg. spam classification or sentiment analysis
### Perceptron
It's a single perceptron with kernel trick
### Support Vector Machine
#### hard margin
Hard margin version is only for doing language processing
#### soft margin
Soft margin version is implemented refer to Platt's paper which introduce SMO algorithm in dealing with KKT condition
### Decision Tree
Using ratio to pick the best spilt feature
### Random Forest
Using bootstrap to pick samples, and pick random features to plant decision trees
## Unsupervised Learning
### Gaussion Mixture Model
Can also deal with semi-supervised learning problems
### K-Means
Powerful in compressing images
### Independent Components Analysis
Powerful in recovering the mixing audios
### Principal Components Analysis
Reducing relevant features
## Deep Learning
### Neural Network
Has three hidden layers, only for binary classification and only sigmoid function as activation func
### Convolutional Neural Network
Architecture can be seen in the comments of the class, for addressing image classification problems
## Todo
-Adaboost


