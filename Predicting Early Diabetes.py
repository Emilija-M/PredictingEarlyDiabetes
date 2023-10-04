#!/usr/bin/env python
# coding: utf-8

# <center> <h1> Predicting Early Diabetes </h1 ></center>

# ### 1. Introduction and Motivation

# $ \;\;\;\;\;\; $ Diabetes is a chronic metabolic disorder that leads to elevated blood glucose levels. More than 415 million people in the world suffer from diabetes, and it is estimated that by 2040, that number will rise to close to 650 million sufferers. A high level of sugar leads to damage to blood vessels, and consequently to kidney failure, as well as to heart attack and stroke. Therefore, there is a need to diagnose diabetes as soon as possible for the most successful treatment. To this end, scientists are looking for a way to create a model that would predict the development of the disease based on available parameters. In this paper, we will present 7 possible forecasting models:
# 
#     1. Logistic regression
#     2. Random Forest
#     3. Naive Bayes Classifier
#     4. Elastic Net
#     5. SVM (Support Vector Machines)
#     6. Neural Networks
#     
# Using these models, early detection of diabetes is possible based on both demographic parameters and laboratory findings and the entire clinical picture that characterizes the patient. In addition to the fact that early detection is important for improving treatment and increasing the chances of a longer and better quality of life, it can also reduce the financial burden caused by the increasing number of patients.
# In general, machine learning is becoming increasingly important due to its extremely promising results in medicine when solving various problems such as classification and prediction.

# #### 1.1 Database
# 
# We use the "Pima Indians Diabetes" database on which we will explain and demonstrate all of the above methods. The database contains 768 observations with 8 predictors and the dependent variable Outcome - an indicator of whether the patient has diabetes. The predictors are:
# 
#   - Pregnancies: number of pregnancies
#   - Glucose: glucose concentration
#   - BloodPressure: blood pressure (mmHg)
#   - SkinThickness: skin thickness on the triceps (mm)
#   - Insulin: insulin level (mu U/ml)
#   - BMI: body mass index (using the formula mass(kg)/(height(m))^2)
#   - DiabetesPedigreeFunction: a function that displays a patient's odds of having diabetes based on family history
#   - Age: age (years)
#   - Outcome: Indicator (0 or 1). 268 out of 768 patients have diabetes (1) and the rest do not (0).

# We load the necessary libraries for work:

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from matplotlib import pyplot
from sklearn.svm import SVC


# ### 2. Application to data

# In[39]:


data = pd.read_csv('diabetes.csv')
data.head()


# Now we can look at some features of our database using the describe and info functions:

# In[40]:


data.describe()


# In[41]:


data.info()
print ('\nMissing values :  ', data.isnull().sum().values.sum())


# We observe that there are no missing values. However, there are zero values that have the same role and which we will replace with e.g. by the median value.

# In[42]:


data.Glucose.replace(0, data['Glucose'].median(), inplace=True)
data.BloodPressure.replace(0, data['BloodPressure'].median(), inplace=True)
data.SkinThickness.replace(0, data['SkinThickness'].median(), inplace=True)
data.Insulin.replace(0, data['Insulin'].median(), inplace=True)
data.BMI.replace(0, data['BMI'].median(), inplace=True)
data.head()


# We show the ratio of two different outcomes in our database:

# In[43]:


sns.countplot(x='Outcome',data=data,palette=['#432371',"#FAAE7B"])


# #### 2.1. Logistic regression
# 
# $ \;\;\;\;\;\; $ In case we need to predict whether the dependent variable will take 0 or 1 instead of the value, we use logistic regression. Specifically, in this paper, we will be interested in whether a person has diabetes - 1 or not - 0. For this reason, we model the probability that the variable Y takes the value 1, if we know what values the other parameters took:
# 
# $$ P\{Y=1|X\} = \pi(x) \in [0,1] $$
# 
# Then we solve the shape regression problem:
# 
# $$ \pi(X) = X\beta, $$
# 
# where $ \beta $ is the vector of coefficients we evaluate, and X is the data matrix.
# Since $ \pi(x) $ must take a value from the interval $ [0,1] $, we introduce the transformation $ g(X\beta) $ which will enable this. There are several different possibilities, but usually the function used is:
# 
# $$ g(x) = \frac{e^{x}}{1+e^{x}}, $$
# 
# and finally we get the model:
# 
# $$ \pi(X) = \frac{e^{X\beta}}{1+e^{X\beta}}. $$
# 
# We evaluate the coefficients using the maximum credibility method, i.e. we seek the maximum of the logarithm of the likelihood function:
# 
# $$ \hat{\beta} = \underset{\beta}{argmax} \sum_{i=1}^{n} (1-y_i)log(1-\pi(x_i)) + y_i log(\pi(x_i)) $$

# We divide the database into training and test set in the ratio 80:20, scale the data and create a model:

# In[160]:


x = data.drop('Outcome',axis=1)
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[153]:


log_model = LogisticRegression()
log_model.fit(x_train, y_train)


# We calculate the standard quality measures on the training set as the mean value of the results obtained by cross-validation:

# In[46]:


kf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy     = np.mean(cross_val_score(log_model, x_train, y_train, cv=kf, scoring='accuracy'))
precision    = np.mean(cross_val_score(log_model, x_train, y_train, cv=kf, scoring='precision'))
recall       = np.mean(cross_val_score(log_model, x_train, y_train, cv=kf, scoring='recall'))
f1score      = np.mean(cross_val_score(log_model, x_train, y_train, cv=kf, scoring='f1'))
rocauc       = np.mean(cross_val_score(log_model, x_train, y_train, cv=kf, scoring='roc_auc'))


# In[47]:


pd.DataFrame({'accuracy'     : [accuracy],
              'precision'    : [precision],
              'recall'       : [recall],
              'f1score'      : [f1score],
              'rocauc'       : [rocauc] })


# Now we make a prediction on the test set:

# In[48]:


predictions = log_model.predict(x_test)


# In[49]:


accuracy_test = accuracy_score(y_test, predictions)
precision_test = precision_score(y_test, predictions)
recall_test = recall_score(y_test, predictions)
f1_test = f1_score(y_test, predictions)
roc_auc_test = roc_auc_score(y_test, predictions)


# In[50]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# We create a confusion matrix and present it graphically:

# In[52]:


conf = confusion_matrix(y_test, predictions)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('true value')


# #### 2.2 Random Forest 
# 
# $ \;\;\;\;\;\; $ Random Forest is a simple machine learning algorithm used for classification and regression. It consists of a large number of individual decision trees which together represent a whole (ensemble). Each individual tree in the Random Forest makes one prediction and the result of the prediction that appears in the highest number is the final prediction of our model.
# In general, ensembles are models that arrive at a prediction by combining several different models. All models have errors that are mutually uncorrelated and it follows that the results will be less biased and less sensitive to different data (they will have less dispersion). There are 2 types of ensemble models:
#     
#   - Simple aggregation (bagging) - which includes Random Forest. This method consists of extracting subsets on which to train the model simultaneously. In the end, we get one prediction from each subset.
#   - Boosting where XGBoost is a typical representative. We also extract subsets, however, model training does not occur in parallel on each subset. Instead, on each subsequent subset, training is performed based on the conclusions obtained from the previous ones.
#   ![1.png](attachment:1.png)

# The Random Forest algorithm works in a couple of steps: first, n subsets are selected on which to train the model, i.e. on which the decision tree is created (each subset creates one tree); further, when creating your tree, features and their values are used when branching. Finally, the most frequent prediction is accepted.
# 
# ![2.jpeg](attachment:2.jpeg)

# Now we will apply this method to our database:

# In[53]:


rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)


# And we recalculate the quality measures on the training and test set.

# In[54]:


kf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy     = np.mean(cross_val_score(rf_model, x_train, y_train, cv=kf, scoring='accuracy'))
precision    = np.mean(cross_val_score(rf_model, x_train, y_train, cv=kf, scoring='precision'))
recall       = np.mean(cross_val_score(rf_model, x_train, y_train, cv=kf, scoring='recall'))
f1score      = np.mean(cross_val_score(rf_model, x_train, y_train, cv=kf, scoring='f1'))
rocauc       = np.mean(cross_val_score(rf_model, x_train, y_train, cv=kf, scoring='roc_auc'))


# In[55]:


pd.DataFrame({'accuracy'     : [accuracy],
              'precision'    : [precision],
              'recall'       : [recall],
              'f1score'      : [f1score],
              'rocauc'       : [rocauc] })


# In[56]:


predictions = rf_model.predict(x_test)


# In[57]:


accuracy_test = accuracy_score(y_test, predictions)
precision_test = precision_score(y_test, predictions)
recall_test = recall_score(y_test, predictions)
f1_test = f1_score(y_test, predictions)
roc_auc_test = roc_auc_score(y_test, predictions)


# In[58]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# We can see that the values of f1score and rocauc on the test set are more favorable for logistic regression.

# In[59]:


conf = confusion_matrix(y_test, predictions)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('real value')


# We can also see which predictors played the biggest role when building the model:

# In[60]:


(pd.Series(rf_model.feature_importances_, index=x.columns)
    .nlargest(8)
    .plot(kind='barh', figsize=[8,4])
    .invert_yaxis())
plt.yticks(size=15)
plt.title('Predictors with the greatest impact', size=20)


# #### 2.3 Naive Bayes classifier
# 
# $ \;\;\;\;\;\; $ The naive Bayes classifier uses Bayes' theorem when predicting:
# 
# $$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$
# 
# If we apply this formula to a classic classification problem, we can write it in the following form:
# 
# $$ P(y|X) = \frac{P(X|y)P(y)}{P(X)}, $$
# 
# where X is the predictor matrix and y is the dependent variable that takes the value 0 or 1.
# In the event that y takes a value from several classes, the class for which the probability P(y|X) is the highest is sought, i.e.
# 
# $$ y_{max} = \underset{y}{argmax}P(y|X) = \underset{y}{argmax} \frac{P(X|y)P(y)}{P(X)} = \underset{y}{argmax} P(X|y)P(y) $$
# 
# The Bayesian classifier assumes that all predictors are mutually independent and have an equal impact on the outcome. That's why he got the name "naive".
# 
# If $ (x_1,x_2,...,x_n) $ denotes the predictors, i.e. the columns of the matrix X, then we can write the mentioned theorem in another way:
# 
# $$ P(y|x_1,x_2,...,x_n) = \frac{P(x_1,x_2,...,x_n|y)P(y)}{P(x_1,x_2,...,x_n)} = \frac{P(x_1|y)P(x_2|y)...P(x_n|y)P(y)}{P(x_1)P(x_2)...P(x_n)}. $$
# 
# This formula is created based on the following:
# 
# $$  P(x_1,x_2,...,x_n|y) = P(x_1|y)P(x_2,...,x_n|y,x_1) = P(x_1|y)P(x_2|y,x_1)P(x_n|y,x_1,x_2,...,x_{n-1}) $$
# 
# and the fact that the predictors are independent.
# 
# Since only the numerator depends on y, we can ignore the denominator. Now it is necessary to evaluate the remaining factors.
# P(y) is equal to the representation of the class in the set of available data, i.e.
# 
# $$ P(y=j) = \frac{\sum_{i=1}^{m} I\{y_i=j\}}{m}, $$
# where m is the number of observations, and j=0,1.
# 
# We distinguish different types of Bayesian classifiers, such as:
# 
# 1. Bernoulli's Naïve Bayesian classifier - used in case the predictors take only true/false values and based on that the score for P(X|y) is obtained.
#   2. Gaussian Naive Bayes classifier - used if the predictors have a continuous distribution function. In that case, we assume that:
#   
#    $$ P(x_i|y) = \frac{1}{\sqrt{2\pi{\sigma_y}^2}}e^{-\frac{(x_i-\mu_y)^2}{2{\sigma_y}^2}} $$

# In[61]:


nb_model = GaussianNB()
nb_model.fit(x_train, y_train)


# In[62]:


kf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy     = np.mean(cross_val_score(nb_model, x_train, y_train, cv=kf, scoring='accuracy'))
precision    = np.mean(cross_val_score(nb_model, x_train, y_train, cv=kf, scoring='precision'))
recall       = np.mean(cross_val_score(nb_model, x_train, y_train, cv=kf, scoring='recall'))
f1score      = np.mean(cross_val_score(nb_model, x_train, y_train, cv=kf, scoring='f1'))
rocauc       = np.mean(cross_val_score(nb_model, x_train, y_train, cv=kf, scoring='roc_auc'))


# In[63]:


pd.DataFrame({'accuracy'     : [accuracy],
              'precision'    : [precision],
              'recall'       : [recall],
              'f1score'      : [f1score],
              'rocauc'       : [rocauc] })


# In[64]:


predictions = nb_model.predict(x_test)


# In[65]:


accuracy_test = accuracy_score(y_test, predictions)
precision_test = precision_score(y_test, predictions)
recall_test = recall_score(y_test, predictions)
f1_test = f1_score(y_test, predictions)
roc_auc_test = roc_auc_score(y_test, predictions)


# In[66]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# In[67]:


conf = confusion_matrix(y_test, predictions)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('real value')


# #### 2.4 Elastic net
# 
# $ \;\;\;\;\;\; $ Elastic net is a type of regularization that combines Ridge and Lasso regularization to improve classical linear regression. As we know, Ridge regularization prevents the occurrence of overfitting of the model by "punishing" large values of the regression coefficients by imposing an additional restriction on the norm of the coefficients, i.e. by minimizing the function:
# 
# $$ (Y-\beta X)^T(Y-\beta X) + c||\beta||_2. $$
# 
# Lasso regularization works in a similar way, except that we use the $ l_1 $ norm for minimization. Lasso can also, unlike Ridge, be used for predictor selection because we can get zero as the score of some coefficients:
# 
# $$ (Y-\beta X)^T(Y-\beta X) + c||\beta||_1. $$
# 
# Elastic net combines these two types and depending on the parameters $ \alpha $ we get:
# 
# $$ (Y-\beta X)^T(Y-\beta X) + \alpha||\beta||_1 +  (1-\alpha)||\beta||_2. $$
# 
# This approach solves some Lasso regularization problems by introducing the $ ||\beta||_2 $ term. For example, in cases where we have a small number of observations (n) and a large number of predictors (p), Lasso will save at most n predictors. Or, if there is a group of highly correlated predictors, Lasso selects one predictor from the group and ignores the others. Also, the quadratic term ensures the convexity of the loss function and thus the uniqueness of the minimum of that function.

# We call the logistic regression again, but this time on the validation set we search for the best model using different values of the penalty parameters (l1, l2 or Elastic net) and c_values (the value of the limiting factor):

# In[82]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
x_tr,x_vl,y_tr,y_vl = train_test_split(x_train,y_train,train_size=0.75,stratify=y_train)
model = LogisticRegression()
penalty = ['l1','l2','elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_vl, y_vl)
print("The highest value of accuracy %f is obtained using the parameter: %s" 
      % (grid_result.best_score_, grid_result.best_params_))


# In[83]:


en_model = LogisticRegression(penalty='l2',C=100)
en_model.fit(x_tr,y_tr)
predictions = en_model.predict(x_test)


# In[84]:


kf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy     = np.mean(cross_val_score(en_model, x_tr, y_tr, cv=kf, scoring='accuracy'))
precision    = np.mean(cross_val_score(en_model, x_tr, y_tr, cv=kf, scoring='precision'))
recall       = np.mean(cross_val_score(en_model, x_tr, y_tr, cv=kf, scoring='recall'))
f1score      = np.mean(cross_val_score(en_model, x_tr, y_tr, cv=kf, scoring='f1'))
rocauc       = np.mean(cross_val_score(en_model, x_tr, y_tr, cv=kf, scoring='roc_auc'))


# In[85]:


pd.DataFrame({'accuracy'     : [accuracy],
              'precision'    : [precision],
              'recall'       : [recall],
              'f1score'      : [f1score],
              'rocauc'       : [rocauc] })


# In[90]:


accuracy_test = accuracy_score(y_test, predictions)
precision_test = precision_score(y_test, predictions)
recall_test = recall_score(y_test, predictions)
f1_test = f1_score(y_test, predictions)
roc_auc_test = roc_auc_score(y_test, predictions)


# In[91]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# In[92]:


conf = confusion_matrix(y_test, predictions)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('real value')


# #### 2.5 SVM 
# 
# $ \;\;\;\;\;\; $ SVM or Support Vector Machine is a linear model that can be used for both classification and regression. His algorithm works by creating a line (hyperplane) that separates the data into classes.
# In a first approximation, the SVM should divide the database into two classes by finding a line that separates them.
# For example, if in the following image we wanted to separate the red squares from the blue circles with a single line, we could do it in an infinite number of ways.
# ![prva.png](attachment:prva.png)
# 

# ![druga.png](attachment:druga.png)

# We are interested in which line out of all the possible ones separating the data is the ideal one to choose. In this case, intuitively the yellow line is a better choice.
# How does SVM find the ideal line in the general case?
# The support vectors are those points that are closest to the line, and the margin is the distance between the support vectors and the line. The ideal line or Optimal hyperplane is the one for which the margin is maximal and must be equally distant from the support vectors of both classes.

# ![treca.png](attachment:treca.png)

# - A more complicated case
# 
# If we have a database that is not this simple, that is, the data within it cannot be separated by drawing only one line, then we have to transform the data so that they can be separated linearly. For example, in the following image we have such a base:
# ![cetvrta.png](attachment:cetvrta.png)

# Here we need to introduce another axis z and it has coordinates:
# $$ z=x^2+y^2 $$
# Therefore, the z-coordinate is the square of the distance of the point from the coordinate origin. When the same data is plotted on the z-axis, it can be seen that the data can now be separated linearly.
# ![peta.png](attachment:peta.png)

# Let the dashed line separating the data have the z value m ($ z=m $).
# Since we have already said that the z axis represents the square of the distance, we get $ m=x^2+y^2 $ or the equation of the circle. We conclude that it is possible to separate the data with a circle when they are given in several dimensions.
# ![sesta.png](attachment:sesta.png)

# When the data in the database is not linearly separable, it is necessary to introduce an additional dimension that will make it so and then find the correct transformation for the optimal line (in this case it was a circle).
# 
# <center> <h4> Hyperplane </h4> </center>
#     
# A hyperplane is a plane in n-dimensional Euclidean space, that is, a subspace of dimension n-1, and it divides the space into two separate parts.
# Example in one dimension:
# - In case the line is our Euclidean space, when we choose a point on that line, it will divide our data into two sets. The point has 0 dimension, which is 1 less than the line. We see that the point is a hyperplane of the line.
# - For two dimensions, a line is a hyperplane
# - For three dimensions, a plane divides three-dimensional space into two parts.
# We see that for every n-dimensional space, there is actually an n-1-dimensional hyperplane.

# We apply the same procedure as before:

# In[93]:


svc_model = SVC(kernel='linear')
svc_model.fit(x_train, y_train)
prediction = svc_model.predict(x_test)


# In[94]:


kf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy     = np.mean(cross_val_score(svc_model, x_train, y_train, cv=kf, scoring='accuracy'))
precision    = np.mean(cross_val_score(svc_model, x_train, y_train, cv=kf, scoring='precision'))
recall       = np.mean(cross_val_score(svc_model, x_train, y_train, cv=kf, scoring='recall'))
f1score      = np.mean(cross_val_score(svc_model, x_train, y_train, cv=kf, scoring='f1'))
rocauc       = np.mean(cross_val_score(svc_model, x_train, y_train, cv=kf, scoring='roc_auc'))


# In[95]:


pd.DataFrame({'accuracy'     : [accuracy],
              'precision'    : [precision],
              'recall'       : [recall],
              'f1score'      : [f1score],
              'rocauc'       : [rocauc] })


# In[96]:


accuracy_test = accuracy_score(y_test, prediction)
precision_test = precision_score(y_test, prediction)
recall_test = recall_score(y_test, prediction)
f1_test = f1_score(y_test, prediction)
roc_auc_test = roc_auc_score(y_test, prediction)


# In[97]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# In[99]:


conf = confusion_matrix(y_test, prediction)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('real value')


# #### 2.6 Neural Networks
# 
# $ \;\;\;\;\;\; $ Neural Networks represent a machine learning model that consists of multiple connected nodes that function in a similar way to the nervous system. One of the main features is the training process, which requires a lot of time and a large database. The idea behind neural networks is to engineer them to mimic learning similar to living things. Nodes (neurons) within the network are organized into layers and there is an input, hidden and output layer.
# Based on the input data, the connection coefficients between the neurons are updated. In this way, the neurons are trained so that they manage to apply the given example to other situations in general. Updating the weight coefficients continues the training process. Each time an output closer to the set value is obtained. After this process, the resulting network is trained and can be applied to other tasks.
# 
# The main terms in the study of neural networks are:
# 
#   - Input
#   - Weights (coefficients)
#   - Bias
#   - Activation function
#   - Output
#   
#   ![3.png](attachment:3.png)
#   
# We have already explained the concepts of input and output layers, as well as the concept of coefficients.
# The entered data is multiplied by the coefficient, the values are summed and then that sum is added with the bias element. The activation function then depending on the value of the sum activates the node. The activated node sends data to the other layers and thus reaches the output layer.

# We import the required libraries and create a neural network model:

# In[110]:


import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[111]:


model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# We calculate quality measures:

# In[112]:


history = model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=0)
_,accuracy = model.evaluate(x_train, y_train)
print(accuracy)


# In[113]:


predictions = (model.predict(x_test) > 0.5).astype(int)
accuracy_test = accuracy_score(y_test, predictions)
precision_test = precision_score(y_test, predictions)
recall_test = recall_score(y_test, predictions)
f1_test = f1_score(y_test, predictions)
roc_auc_test = roc_auc_score(y_test, predictions)


# In[114]:


pd.DataFrame({'accuracy'     : [accuracy_test],
              'precision'    : [precision_test],
              'recall'       : [recall_test],
              'f1score'      : [f1_test],
              'rocauc'       : [roc_auc_test] })


# In[115]:


conf = confusion_matrix(y_test, predictions)
plt.figure(figsize = [5,5])
sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d', 
            xticklabels=['no diabetes', 'diabetes'], yticklabels=['no diabetes', 'diabetes']);
plt.xlabel('prediction')
plt.ylabel('real value')


# As we have already said, the more the model is trained, the better it is, and accordingly we notice how the loss function decreases and the accuracy increases:

# In[116]:


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Loss function')
plt.plot(np.arange(0, 150), history.history['loss'], label='train')
plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(np.arange(0, 150), history.history['accuracy'])
plt.show()


# ### 3. Application of the model on another database
# 
# Now we will apply the described models on another database. Data were obtained from 150 hospitals in the period from 1999 to 2008. Each row corresponds to one patient, and predictors can be grouped into categories. This is patient demographic information, measurements taken at the hospital, and diabetes medications the patient is taking.

# In[307]:


diabetes = pd.read_csv('diabetic_data.csv')


# In[308]:


diabetes


# After reviewing the database, we conclude that based on the number of missing data, we can remove the columns weight, payer_code and medical_specialty because we believe they will not affect the creation of the model. Also, the predictors encounter_id and patient_nbr are of identification character, and we can ignore them. The predictors examide and citoglipton have only one value, so we can also exclude them. We change the missing values in the race column with the mode ('Caucasian'). In the columns diag_1, diag_2 and diag_3 there is data about the diagnosis, which is partly covered by the information in the column number_diagnoses, and for the sake of simplification, we will leave them out.

# In[309]:


print(sum(diabetes.weight == '?'))


# In[310]:


print(sum(diabetes.payer_code == '?'))


# In[311]:


print(sum(diabetes.medical_specialty == '?'))


# In[312]:


diabetes['examide'].value_counts()


# In[313]:


diabetes['citoglipton'].value_counts()


# In[314]:


diabetes = diabetes.loc[:, diabetes.columns != 'weight']
diabetes = diabetes.loc[:, diabetes.columns != 'payer_code']
diabetes = diabetes.loc[:, diabetes.columns != 'medical_specialty']
diabetes = diabetes.loc[:, diabetes.columns != 'examide']
diabetes = diabetes.loc[:, diabetes.columns != 'citoglipton']
diabetes = diabetes.loc[:, diabetes.columns != 'encounter_id']
diabetes = diabetes.loc[:, diabetes.columns != 'patient_nbr']
diabetes = diabetes.loc[:, diabetes.columns != 'diag_1']
diabetes = diabetes.loc[:, diabetes.columns != 'diag_2']
diabetes = diabetes.loc[:, diabetes.columns != 'diag_3']
diabetes.head()


# In[315]:


diabetes['race'].value_counts()


# In[316]:


diabetes.race.replace('?', 'Caucasian', inplace=True)


# In[317]:


diabetes['race'].value_counts()


# In[318]:


diabetes['gender'].value_counts()


# We delete the three missing fields we noticed:

# In[319]:


diabetes.gender.replace('Unknown/Invalid', np.nan, inplace=True)
diabetes=diabetes.dropna(subset=['gender'])


# We continue to organize the data. We code all predictors that have two value levels (gender - Male/Female, change - No/Ch, diabetesMed - No/Yes) with 0 and 1:

# In[320]:


diabetes.gender.replace('Male', 0, inplace = True)
diabetes.gender.replace('Female', 1, inplace = True)
diabetes.change.replace('No', 0, inplace = True)
diabetes.change.replace('Ch', 1, inplace = True)
diabetes.diabetesMed.replace('No', 0, inplace = True)
diabetes.diabetesMed.replace('Yes', 1, inplace = True)


# In[321]:


diabetes['gender'].value_counts()


# In[322]:


diabetes['change'].value_counts()


# In[323]:


diabetes['diabetesMed'].value_counts()


# To encode the race variable, we use One hot encoding and place the resulting columns in a new database. We transform the age variable so that instead of the interval we take its middle.

# In[324]:


one_hot_enc = pd.get_dummies(diabetes.race).replace({False:0, True:1})
print(one_hot_enc)


# In[325]:


df = diabetes.join(one_hot_enc)


# In[326]:


df = df.loc[:, df.columns != 'race']
df.head()


# In[327]:


age_id = {'[0-10)':5,
          '[10-20)':15,
          '[20-30)':25,
          '[30-40)':35,
          '[40-50)':45,
          '[50-60)':55,
          '[60-70)':65,
          '[70-80)':75,
          '[80-90)':85,
          '[90-100)':95}
df['age_group'] = df.age.replace(age_id) 
df = df.loc[:, df.columns != 'age']
df


# In[328]:


print(df. columns)


# The rest of the categorical variables can be transformed into numerical ones because the data are of an ordinal character - ie. we can assign them the values 0, 1, 2 and 3.

# In[329]:


df.max_glu_serum.replace('>300', 3, inplace = True)
df.max_glu_serum.replace('>200', 2, inplace = True)
df.max_glu_serum.replace('Norm', 1, inplace = True)
df.max_glu_serum.replace('None', 0, inplace = True)
df['max_glu_serum'].value_counts()
df


# In[330]:


df.A1Cresult.replace('>8', 3, inplace = True)
df.A1Cresult.replace('>7', 2, inplace = True)
df.A1Cresult.replace('Norm', 1, inplace = True)
df.A1Cresult.replace('None', 0, inplace = True)
df['A1Cresult'].value_counts()


# In[331]:


df.rename(columns = {'glyburide-metformin':'glyburide_metformin'}, inplace = True)
df.rename(columns = {'glipizide-metformin':'glipizide_metformin'}, inplace = True)
df.rename(columns = {'glimepiride-pioglitazone':'glimepiride_pioglitazone'}, inplace = True)
df.rename(columns = {'metformin-rosiglitazone':'metformin_rosiglitazone'}, inplace = True)
df.rename(columns = {'metformin-pioglitazone':'metformin_pioglitazone'}, inplace = True)


# In[332]:


meds = [df.metformin, df.repaglinide, df.nateglinide, df.chlorpropamide, df.glimepiride, df.acetohexamide, 
        df.glipizide, df.glyburide, df.tolbutamide, df.pioglitazone, df.rosiglitazone, df.acarbose, 
        df.miglitol, df.troglitazone, df.tolazamide, df.insulin, df.glyburide_metformin, df.glipizide_metformin, 
        df.glimepiride_pioglitazone, df.metformin_rosiglitazone, df.metformin_pioglitazone]
for x in meds:
    x.replace('Up', 3, inplace = True)
    x.replace('Steady', 2, inplace = True)
    x.replace('Down', 1, inplace = True)
    x.replace('No', 0, inplace = True)


# We code the values of the dependent variable with 1 if the patient was readmitted and 0 if not.

# In[333]:


df.readmitted.replace('>30', 1, inplace = True)
df.readmitted.replace('NO', 0, inplace = True)
df.readmitted.replace('<30', 1, inplace = True)
df


# We then check for missing data:

# In[334]:


df.isnull().sum()


# Since predictors max_glu_serum and A1Cresult have too many missing data, we conclude that is best to exclude them from further analysis.

# In[335]:


df = df.loc[:, df.columns != 'max_glu_serum']
df = df.loc[:, df.columns != 'A1Cresult']
df.isnull().sum()


# Based on the obtained data, we make models again and calculate their quality measures:

# We check the predictor types to make sure there are no unwanted types that would make it difficult for us to build models over them.

# In[336]:


df.dtypes


# In[337]:


x = df.drop('readmitted',axis=1)
y = df['readmitted']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[347]:


logit = LogisticRegression(solver='liblinear')
logit.fit(x_train, y_train)
y_pred = logit.predict(x_train)
accuracy_log     = accuracy_score(y_train, y_pred)
precision_log    = precision_score(y_train, y_pred)
recall_log       = recall_score(y_train, y_pred)
f1score_log      = f1_score(y_train, y_pred)
rocauc_log       = roc_auc_score(y_train, y_pred)


# In[339]:


gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_train)
accuracy_nb     = accuracy_score(y_train, y_pred)
precision_nb    = precision_score(y_train, y_pred)
recall_nb       = recall_score(y_train, y_pred)
f1score_nb      = f1_score(y_train, y_pred)
rocauc_nb       = roc_auc_score(y_train, y_pred)


# In[340]:


randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_train)
accuracy_rf     = accuracy_score(y_train, y_pred)
precision_rf    = precision_score(y_train, y_pred)
recall_rf       = recall_score(y_train, y_pred)
f1score_rf      = f1_score(y_train, y_pred)
rocauc_rf       = roc_auc_score(y_train, y_pred)


# In[341]:


from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(x_train,y_train,test_size=0.2)
n_estimators = 10
svc = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), 
                                            max_samples=1.0 / n_estimators, n_estimators=n_estimators),
                          n_jobs=-1)
svc.fit(x_test_1, y_test_1)
y_pred = svc.predict(x_test_1)
accuracy_sv     = accuracy_score(y_test_1, y_pred)
precision_sv    = precision_score(y_test_1, y_pred)
recall_sv       = recall_score(y_test_1, y_pred)
f1score_sv      = f1_score(y_test_1, y_pred)
rocauc_sv       = roc_auc_score(y_test_1, y_pred)


# In[348]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


x_tr,x_vl,y_tr,y_vl = train_test_split(x_train,y_train,train_size=0.75,stratify=y_train)
model = LogisticRegression(solver='liblinear')
penalty = ['l1','l2','elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_vl, y_vl)
print("The highest value of accuracy %f is obtained using parameters: %s" 
      % (grid_result.best_score_, grid_result.best_params_))


# In[349]:


en = LogisticRegression(penalty='l2',C=0.01,solver='liblinear')
en.fit(x_tr,y_tr)
y_pred = en.predict(x_train)
accuracy_en     = accuracy_score(y_train, y_pred)
precision_en    = precision_score(y_train, y_pred)
recall_en       = recall_score(y_train, y_pred)
f1score_en      = f1_score(y_train, y_pred)
rocauc_en       = roc_auc_score(y_train, y_pred)


# In[365]:


model = Sequential()
model.add(Dense(12, input_shape=(41,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=0)
y_pred = (model.predict(x_train) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_train, y_pred)
precision_nn = precision_score(y_train, y_pred)
recall_nn = recall_score(y_train, y_pred)
f1_nn = f1_score(y_train, y_pred)
roc_auc_nn = roc_auc_score(y_train, y_pred)


# In[366]:


df_models = pd.DataFrame(
    {'model'        : ["Logistic regression", "Naive Bayes", "Random Forest", "SVM", "Elastic Net", "Neural Networks"],
     'accuracy'     : [accuracy_log, accuracy_nb, accuracy_rf, accuracy_sv, accuracy_en, accuracy_nn],
     'precision'    : [precision_log, precision_nb, precision_rf, precision_sv, precision_en, precision_nn],
     'recall'       : [recall_log, recall_nb, recall_rf, recall_sv, recall_en, recall_nn],
     'f1score'      : [f1score_log, f1score_nb, f1score_rf, f1score_sv, f1score_en, f1_nn],
     'rocauc'       : [rocauc_log, rocauc_nb, rocauc_rf, rocauc_sv, rocauc_en, roc_auc_nn]})
df_models


# We also draw a graphic display of the F1 score value and the ROC curve:

# In[367]:


import sklearn
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(2, 1, figsize=(16, 18))

ax[0].bar(df_models.model, df_models.f1score)
ax[0].set_title('F1-score')

model_name = [gnb, logit, randomforest, svc, en]

for i in range(5):
    y_pred = model_name[i].predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, lw=3, label=df_models.model[i] + ' (area = %0.3f)' % sklearn.metrics.auc(fpr, tpr))

y_pred = model.predict(x_test) 
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, lw=3, label='Neural Networks (area = %0.3f)' % sklearn.metrics.auc(fpr, tpr))

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Display of ROC krive', fontsize=17)
plt.legend(loc='lower right', fontsize=13)
plt.show()


# ### 4. Conclusion
# 
# We started from a simpler base where we first introduced the models we work with and briefly described them. We applied each model to the data and described its quality through several parameters, both on the training and on the test set. In the second part, we applied the mentioned models to the proposed database and compared their accuracies in the table. It was clear that the Random Forest algorithm gives convincingly the best results (which is also suggested in the literature), and the naive Bayesian classifier the worst. We further confirm this conclusion with the graphic display at the very end.

# ### 5. Literature
# 
#  - http://cs229.stanford.edu/proj2017/final-reports/5244347.pdf
#  - https://www.dropbox.com/s/rgrfqcpylq9luoi/Statisticki_Softver_3_2019_2020.pdf?dl=0
#  - https://jnyh.medium.com/building-a-machine-learning-classifier-model-for-diabetes-4fca624daed0
#  - http://enastava.matf.bg.ac.rs/pluginfile.php/27875/mod_resource/content/3/ansambli%20%282%29-pages-deleted.pdf
#  - https://towardsdatascience.com/understanding-random-forest-58381e0602d2
#  - https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
#  - https://en.wikipedia.org/wiki/Elastic_net_regularization
#  - https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
#  - http://enastava.matf.bg.ac.rs/pluginfile.php/27874/mod_resource/content/1/SVM%20%285%29.pdf
#  - https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
#  - https://www.ibm.com/docs/en/spss-modeler/saas?topic=models-how-svm-works
#  - http://enastava.matf.bg.ac.rs/pluginfile.php/28141/mod_resource/content/1/FCNN.pdf
#  - https://towardsdatascience.com/build-the-artificial-intelligence-for-detecting-diabetes-using-neural-networks-and-keras-89962097f0b0
