
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[92]:

data = pd.read_csv('combined_res.csv')

data_res = pd.read_csv('combined_res_ratio.csv')


# In[93]:

data.shape


# In[94]:

df.head()


# In[95]:

df = data.merge(data_res, on=['fyear','gvkey'],how = 'inner')


# In[96]:

dep_vars = ['encoder_error','lstm_error','encoder_error_ratio','lstm_error_ratio']


# In[97]:

from sklearn.linear_model import LogisticRegression


# In[98]:

lr = LogisticRegression()


# In[14]:

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score


# In[15]:

scoring = ['precision_macro', 'recall_macro']


# In[18]:

scores = cross_validate(lr, df[dep_vars], df['truth_x'], scoring=scoring,
                         cv=5, return_train_score=False)


# In[20]:

from sklearn.model_selection import cross_val_predict


# In[21]:

predicted = cross_val_predict(lr, df[dep_vars], df['truth_x'], cv=10)


# In[22]:

from sklearn import metrics


# In[12]:

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[26]:

classification_report(df['truth_x'], predicted)


# In[ ]:




# In[ ]:




# In[27]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[33]:

scores=[]
for val in range(1,41):
    print (val)
    model = RandomForestClassifier(n_estimators=val,class_weight='balanced')
    validated = cross_val_score(model, df[dep_vars], df['truth_x'], cv=3,scoring='f1')
    scores.append(validated)


# In[31]:

import seaborn as sns


# In[34]:

model = RandomForestClassifier(n_estimators=5,class_weight='balanced')
model.fit(df[dep_vars], df['truth_x'])


# In[35]:

preds = model.predict(df[dep_vars])


# In[39]:

classification_report(df['truth_x'], preds)


# In[40]:

conf_matrix = confusion_matrix(df['truth_x'], preds)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




# In[99]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[dep_vars], df['truth_x'],
                                                    stratify=df['truth_x'], 
                                                    test_size=0.2)


# In[42]:

X_train.shape


# In[43]:

X_test.shape


# In[100]:

model = RandomForestClassifier(n_estimators=60,max_depth = 7, class_weight={False:0.1, True:25})
model.fit(X_train,y_train)


# In[101]:

preds = model.predict(X_test)


# In[102]:

classification_report(y_test, preds)


# In[103]:

conf_matrix = confusion_matrix(y_test, preds)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




# In[104]:

total_preds = model.predict(df[dep_vars])


# In[105]:

classification_report(df['truth_x'], total_preds)


# In[106]:

conf_matrix = confusion_matrix(df['truth_x'], total_preds)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[107]:

model.predict_proba(df[dep_vars])


# In[108]:

probs = model.predict_proba(df[dep_vars])


# In[212]:

probs[:,1].sum()


# In[213]:

probs[:,0].sum()


# In[109]:

df['pred_prob'] = probs[:,1]


# In[110]:

df.columns


# In[111]:

df = df.drop(['truth_y'],axis=1)


# In[112]:

df.to_csv('results_just_testing_data.csv',index=False)


# In[ ]:




# In[ ]:




# In[2]:

data = pd.read_csv('all_combined_res.csv')

data_res = pd.read_csv('all_combined_res_ratio.csv')


# In[3]:

df = data.merge(data_res, on=['fyear','gvkey'],how = 'inner')


# In[4]:

dep_vars = ['encoder_error_ratio_x','lstm_error_ratio_x','encoder_error_ratio_y','lstm_error_ratio_y']


# In[5]:

df.head()


# In[6]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[7]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[dep_vars], df['truth_x'],
                                                    stratify=df['truth_x'], 
                                                    test_size=0.2)


# In[8]:

X_train.shape


# In[49]:

model = RandomForestClassifier(n_estimators=10,max_depth = 6, class_weight={False:0.1,True:15})
model.fit(X_train,y_train)


# In[50]:

preds = model.predict(X_test)


# In[51]:

classification_report(y_test, preds)


# In[52]:

total_preds = model.predict(df[dep_vars])


# In[53]:

import seaborn as sns
conf_matrix = confusion_matrix(df['truth_x'], total_preds)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




# In[60]:

model = RandomForestClassifier(n_estimators=4,max_depth = 6, class_weight={True:200, False:1})
model.fit(df[dep_vars],df['truth_x'])


# In[61]:

total_preds = model.predict(df[dep_vars])


# In[62]:

conf_matrix = confusion_matrix(df['truth_x'], total_preds)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[82]:

probs = model.predict_proba(df[dep_vars])


# In[84]:

df['pred_prob'] = probs[:,1]


# In[88]:

df = df.rename(columns={'truth_x': 'misstated', 'encoder_error_ratio_x': 'encoder_error',
                       'lstm_error_ratio_x':'lstm_error','encoder_error_ratio_y':'encoder_error_ratio',
                        'lstm_error_ratio_y':'lstm_error_ratio'
                       })


# In[89]:

df=df.drop(['truth_y'],axis=1)


# In[91]:

df.to_csv("results.csv",index=False)


# In[ ]:



