#!/usr/bin/env python
# coding: utf-8

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter
from yellowbrick.classifier import ROCAUC
from yellowbrick.features import Rank1D, Rank2D
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, cross_validate, train_test_split, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[17]:


data = pd.read_csv('E:/data.csv')
data.head()


# # Data Preprocessing

# In[18]:


data.sample(10)


# In[19]:


data.shape


# In[20]:


sns.countplot(data.SARSCov)
plt.show()


# In[21]:


cols=data.columns
cols


# In[22]:


data.info()


# In[23]:


data.describe()


# In[24]:


missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[25]:


print('Train columns with null values:\n', data.isnull().sum())
print("-"*10)

data.describe(include = 'all')


# # Clean Data

# In[26]:


data_cleaner = [data]


# In[27]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[28]:


print('Train columns with null values:\n', data.isnull().sum())
print("-"*10)

data.describe(include = 'all')


# In[29]:


# percentage of missing data per category
total = data.isnull().sum().sort_values(ascending=False)
percent_total = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]
missing_data


# In[30]:


plt.figure(figsize=(9,6))
sns.set(style="whitegrid")
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)
plt.title('Percentage of missing data by feature')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.show()


# In[31]:


#preview data again
data.info()

#data.sample(10)


# In[32]:


# plot histogram to see the distribution of the data
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)
plt.show()


# In[33]:


sns.countplot(x='SARSCov',data=data)
plt.show()
cases = data.SARSCov.value_counts()
print(f"There are {cases[0]} patients without COVID-19 disease and {cases[1]} patients with the disease")


# In[34]:


positive_cases = data[data['SARSCov'] == 1]
plt.figure(figsize=(15,6))
sns.countplot(x='Age',data = positive_cases, hue = 'SARSCov', palette='husl')
plt.show()


# In[35]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(), annot = True)
plt.show()


# # Data Profiling

# In[36]:


from pandas_profiling import ProfileReport
import pandas_profiling as pdp

#https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html
    
profile = ProfileReport(data, title='Pandas Profiling Report', minimal=True,progress_bar=False,      
    missing_diagrams={
          'heatmap': False,
          'dendrogram': False,
      } )
profile


# In[37]:


features = ['Patient No', 'Age', 'Gender', 'Platelet Distribution Width(PDW)',
       'Mean Platelet Volume (MPV)', 'Platelet (PLT)',
       'Erythrocyte Distribution Width (RDW)',
       'Average Erythrocyte Volume (MCV)', 'Hemoglobin (HGB)',
       'Lymphocyte Count', 'Neutrophil Count', 'Leukocyte (WBC)',
       'CRP (Turbidimetric)', 'LDH', 'Creatine Kinase(CK) (Serum/Plazma)',
       'Creatine kinase-MB (CK-MB) ', 'Troponin I', 'Kreatinin', 'AST', 'ALT',
       'D-dimer, quantitative', 'Procalcitonin', 'PT (INR)', 'PT (sn)', 'APTT']

X = pd.DataFrame(data=data, columns=features)
y = pd.DataFrame(data=data, columns=["SARSCov"])
y = y.astype(int)
y = y-1 
X.head()


# # Features
# #Basic Feature Stats

# In[38]:


from scipy.stats import probplot,skew

for i in cols:
    fig, axes = plt.subplots(1, 3, figsize=(20,4))
    sns.distplot(data[i],kde=False, ax=axes[0])
    sns.boxplot(data[i], ax=axes[1])
    probplot(data[i], plot=axes[2])
    skew_val=round(data[i].skew(), 1)
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[0].set_title(i + " | Distplot")
    axes[1].set_title(i + " | Boxplot")
    axes[2].set_title(i + " | Probability Plot - Skew: "+str(skew_val))
    plt.show()


# # Correlation Heatmap

# In[39]:


def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show()


# In[40]:


correlation_heatmap(data)


# # BorutaPy feature selection techniques

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# In[42]:


#define the features
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)
 
# find all relevant features
feat_selector.fit(X, y)


# In[43]:


# show the most important features
most_important = data.columns[:-1][feat_selector.support_].tolist()
most_important


# In[44]:


# select the top 6 features
top_features = data.columns[:-1][feat_selector.ranking_ <=20].tolist()
top_features


# In[45]:


#import statsmodels.api as sm


# In[ ]:


#X_top = data[top_features]
#y = data['SARSCov']


# In[ ]:


#res = sm.Logit(y,X_top).fit()
#res.summary()


# In[ ]:


#params = res.params
#conf = res.conf_int()
#conf['Odds Ratio'] = params
#conf.columns = ['5%', '95%', 'Odds Ratio']
#print(np.exp(conf))


# # Models and predictions

# In[46]:


x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values


# In[47]:


x


# In[48]:


y


# # Splitting the dataset into the Training set and Test set 

# In[49]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)


# In[50]:


print("Number transactions x_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions x_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# # Feature Scaling

# In[51]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # Handling Imbalance data using SMOTE 

# In[52]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


# In[53]:


X = data[top_features]
y = data.iloc[:,-1]


# In[54]:


# the numbers before SMOTE
num_before = dict(Counter(y))

#perform SMOTE

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X, y)


#the numbers after SMOTE
num_after =dict(Counter(y_smote))


# In[55]:


print(num_before, num_after)


# In[56]:


labels = ["Negative Cases","Positive Cases"]
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.barplot(labels, list(num_before.values()))
plt.title("Numbers Before Balancing")
plt.subplot(1,2,2)
sns.barplot(labels, list(num_after.values()))
plt.title("Numbers After Balancing")
plt.show()


# # Splitting data to Training and Testing set

# In[57]:


# new dataset
new_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
new_data.columns = ['Patient No', 'Age', 'Gender', 'Platelet Distribution Width(PDW)',
       'Mean Platelet Volume (MPV)', 'Platelet (PLT)',
       'Erythrocyte Distribution Width (RDW)',
       'Average Erythrocyte Volume (MCV)', 'Hemoglobin (HGB)',
       'Lymphocyte Count', 'Neutrophil Count', 'Leukocyte (WBC)',
       'CRP (Turbidimetric)', 'LDH', 'Creatine Kinase(CK) (Serum/Plazma)',
       'Creatine kinase-MB (CK-MB) ', 'Troponin I', 'Kreatinin', 'AST', 'ALT',
       'D-dimer, quantitative', 'Procalcitonin', 'PT (INR)', 'PT (sn)', 'APTT', 'SARSCov']
new_data.head()


# In[58]:


X_new = new_data[top_features]
y_new= new_data.iloc[:,-1]
X_new.head()


# In[59]:


# split the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=.2,random_state=42)


# # Feature Scaling

# In[60]:


from sklearn.preprocessing import StandardScaler


# In[61]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# # Models
# The four algorithms that will be used are:
# 
# Logistic Regression
# k-Nearest Neighbours
# Decision Trees
# Support Vector Machine

# # https://www.kaggle.com/code/amayomordecai/heart-disease-risk-prediction-machine-learning/notebook

# # Logistic Regression

# In[62]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve


# In[63]:


# search for optimun parameters using gridsearch
params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=10)


# In[64]:


#train the classifier
logistic_clf.fit(X_train,y_train)

logistic_clf.best_params_


# In[65]:


#make predictions
logistic_predict = logistic_clf.predict(X_test)


# In[66]:


log_accuracy = accuracy_score(y_test,logistic_predict)
print(f"Using logistic regression we get an accuracy of {round(log_accuracy*100,2)}%")


# In[67]:


cm=confusion_matrix(y_test,logistic_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Greens")


# In[68]:


print(classification_report(y_test,logistic_predict))


# In[69]:


logistic_f1 = f1_score(y_test, logistic_predict)
print(f'The f1 score for logistic regression is {round(logistic_f1*100,2)}%')


# In[70]:


# ROC curve and AUC 
probs = logistic_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
log_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(7,4))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(log_auc,3)}")
plt.show()


# # k-Nearest Neighbours

# In[71]:


from sklearn.neighbors import KNeighborsClassifier


# In[72]:


# search for optimun parameters using gridsearch
params= {'n_neighbors': np.arange(1, 10)}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = params, 
                           scoring = 'accuracy', cv = 10, n_jobs = -1)
knn_clf = GridSearchCV(KNeighborsClassifier(),params,cv=3, n_jobs=-1)


# In[73]:


# train the model
knn_clf.fit(X_train,y_train)
knn_clf.best_params_ 


# In[74]:


# predictions
knn_predict = knn_clf.predict(X_test)


# In[75]:


#accuracy
knn_accuracy = accuracy_score(y_test,knn_predict)
print(f"Using k-nearest neighbours we get an accuracy of {round(knn_accuracy*100,2)}%")


# In[76]:


cm=confusion_matrix(y_test,knn_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Greens")


# In[77]:


print(classification_report(y_test,knn_predict))


# In[78]:


knn_f1 = f1_score(y_test, knn_predict)
print(f'The f1 score for K nearest neignbours is {round(knn_f1*100,2)}%')


# In[79]:


# ROC curve and AUC 
probs = knn_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
knn_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(7,4))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(knn_auc,3)}")
plt.show()


# # Decision Trees

# In[80]:


from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(random_state=7)


# In[81]:


# grid search for optimum parameters
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)


# In[82]:


# train the model
tree_clf.fit(X_train,y_train)
tree_clf.best_params_ 


# In[83]:


# predictions
tree_predict = tree_clf.predict(X_test)


# In[84]:


#accuracy
tree_accuracy = accuracy_score(y_test,tree_predict)
print(f"Using Decision Trees we get an accuracy of {round(tree_accuracy*100,2)}%")


# In[85]:


cm=confusion_matrix(y_test,tree_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Greens")


# In[86]:


print(classification_report(y_test,tree_predict))


# In[87]:


tree_f1 = f1_score(y_test, tree_predict)
print(f'The f1 score Descision trees is {round(tree_f1*100,2)}%')


# In[88]:


# ROC curve and AUC 
probs = tree_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
tree_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(7,4))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(tree_auc,3)}")
plt.show()


# # Support Vector Machine

# In[89]:


from sklearn.svm import SVC


# In[90]:


#grid search for optimum parameters
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svm_clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=10)


# In[91]:


# train the model
svm_clf.fit(X_train,y_train)
svm_clf.best_params_ 


# In[92]:


# predictions
svm_predict = svm_clf.predict(X_test)


# In[93]:


#accuracy
svm_accuracy = accuracy_score(y_test,svm_predict)
print(f"Using SVM we get an accuracy of {round(svm_accuracy*100,2)}%")


# In[94]:


cm=confusion_matrix(y_test,svm_predict)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Greens")


# In[95]:


print(classification_report(y_test,svm_predict))


# In[96]:


svm_f1 = f1_score(y_test, svm_predict)
print(f'The f1 score for SVM is {round(svm_f1*100,2)}%')


# In[97]:


# ROC curve and AUC 
probs = svm_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
svm_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(7,4))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(svm_auc,3)}")
plt.show()


# # Model Comparison

# In[98]:


comparison = pd.DataFrame({
    "Logistic regression":{'Accuracy':log_accuracy, 'AUC':log_auc, 'F1 score':logistic_f1},
    "K-nearest neighbours":{'Accuracy':knn_accuracy, 'AUC':knn_auc, 'F1 score':knn_f1},
    "Decision trees":{'Accuracy':tree_accuracy, 'AUC':tree_auc, 'F1 score':tree_f1},
    "Support vector machine":{'Accuracy':svm_accuracy, 'AUC':svm_auc, 'F1 score':svm_f1}
}).T


# In[99]:


comparison


# In[100]:


fig = plt.gcf()
fig.set_size_inches(15, 15)
titles = ['AUC','Accuracy','F1 score']
for title,label in enumerate(comparison.columns):
    plt.subplot(2,2,title+1)
    sns.barplot(x=comparison.index, y = comparison[label], data=comparison)
    plt.xticks(fontsize=9)
    plt.title(titles[title])
plt.show()


# # Cross validation score of the best model

# In[101]:


from sklearn.model_selection import cross_val_score


# In[102]:


cv_results = cross_val_score(svm_clf, X, y, cv=5) 

print (f"Cross-validated scores {cv_results}")
print(f"The Cross Validation accuracy is: {round(cv_results.mean() * 100,2)}%")


# In[ ]:




