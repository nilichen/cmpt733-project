
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats


# In[ ]:

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


# In[ ]:

data = pd.read_csv("final_data.csv")


# In[ ]:

balance_variables = ['aco','acominc','acqcshi','acqgdwl','acqintan','acqlntal','act','ao','ap','arc','at',
                    'ceq','che','cld2','cld3','cld4','cld5','clt','dcom','dcpstk','dcvt','dlc','dltt','dm',
                    'dpacls','drc','drlt','esopct','esopdlt','fatb','fate','fatl','fatn','fato','fea','fel','gdwl',
                    'gwo','icapt','intan','intc','invfg','invofs','invrm','invt','invwip','iseq','iseqc','iseqm','isfi',
                    'isgr','isgu','itcb','lct','lcuacu','lifr','lt','mibt','npat','pll','ppegt','ppenb',
                    'ppenc','ppenls','ppennr','ppent','ppevo','ppevr','pstkc','pvcl','pvo','pvpl','pvt','re',
                    'rect','reuna','rll','rvlrv','rvti','sco','stbo','stio','stkco','teq','tfva','tfvl','tlcf',
                    'transa','tsa','tstkc','tstkp','txdb','txdba','txdbca','txdbcl','txr','txtubxintbs',
                    'vpac','vpo','wcap','xacc','xpp']

incoming_variables = ['amgw','cga','ci','cimii','cogs','do','dp','dvc','ebit','ebitda','fca','gdwlam','gdwlia','gla','gp',
                     'hedgegl','idiis','idilc','idit','llrci','llwoci','ni','niint','niintpfc','niintpfp','nopio',
                     'nrtxt','oiadp','oibdp','opini','opiti','pcl','pi','pifo','pnca','pncia','rca','rdip',
                     'revt','rmum','spi','sppiv','sret','tfvce','txc','txfed','txfo','txp','txt','txtubxintis',
                     'wda','xad','xagt','xdp','xeqo','xi','xido','xinst','xint','xintd','xintopt','xlr',
                     'xopr','xoprar','xpr','xrd','xrent','xs','xsga','xstf','xstfo','xstfws','xt']

opinion_variables = ['auop','auopic','bkvlps','cshi','dvpa','dvpsp_f','emol','emp','epsfi','epspi','exre','ipodate',
                    'mkvalt','ob','opeps','oprepsx','optca','optex','optexd','optfvgr','optgr','optprcca',
                    'optprcex','optprcey','optprcgr','optprcwa','optvol','prcc_f','prch_f','prcl_f','rank',
                    'rpag','sale','salepfc','salepfp','spce','spcindcd','spcseccd','ssnp']

cashflow_variables = ['capx','cdvc','chech','depc','dv','dvpdp','fincf','fopt','invch','ivaeq','ivao','ivncf','ivst',
                     'oancf','prstkcc','prstkpc','scstkc','seq','siv','sppe','spstkc','tdc','txpd',
                     'wcapc','wcapch']
other_vars = ['opiti', 'csho', 'dvp', 'dvpd', 'txpd', 'intpn', 'txp', 'ivst',
              'ivpt', 'dlc', 'pstk', 'emp', 'ob', 'capx', 'bkvlps', 'ivaeq', 'ivao', 'ap']

ratio_cols = ['current', 'quick', 'cash', 'debt2asset', 'debt2capital', 'debt2equity',
              'financial_lev', 'gross_profit_margin', 'pretax_margin', 'net_profit_margin',
              'ROA', 'ROTC', 'ROE', 'ROCE', 'p2e', 'p2bv', 'ebit2interest', 'ebitda2interest',
              'ROC', 'cash_flow2debt', 'cash_flow2rev', 'CROA', 'CROE', 'debt_cov', 'WC_acc',
              'rsst_acc', 'ch_res', 'ch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa',
              'ch_fcf', 'ch_emp', 'ch_bocklog', 'exfin', 'bm', 'ep']

all_variables = list(set(balance_variables + incoming_variables + opinion_variables + cashflow_variables +other_vars))
all_variables.append('misstated')


# In[ ]:

ratio_cols = ['current', 'quick', 'cash', 'debt2asset', 'debt2capital', 'debt2equity',
              'financial_lev', 'gross_profit_margin', 'pretax_margin', 'net_profit_margin',
              'ROA', 'ROTC', 'ROE', 'ROCE', 'p2e', 'p2bv', 'ebit2interest', 'ebitda2interest',
              'ROC', 'cash_flow2debt', 'cash_flow2rev', 'CROA', 'CROE', 'debt_cov', 'WC_acc',
              'rsst_acc', 'ch_res', 'ch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa',
              'ch_fcf', 'ch_emp', 'ch_bocklog', 'exfin', 'bm', 'ep']


# In[ ]:

from sklearn.preprocessing import StandardScaler

for i in all_variables:
    if i!='misstated':
        data[i] = StandardScaler().fit_transform(data[i].values.reshape(-1, 1))


# In[ ]:

#all_variables = incoming_variables
#all_variables.append('misstated')
X_train, X_test = train_test_split(data[all_variables], test_size=0.2, random_state=123)
print(len(X_train))
X_train = X_train[X_train['misstated'] == False]
X_train = X_train.drop(['misstated'], axis=1)
print(len(X_train))

y_test = X_test['misstated']
X_test = X_test.drop(['misstated'], axis=1)

X_train = X_train.values
X_test = X_test.values

X_train.shape


# In[ ]:

#from sklearn.decomposition import PCA


# In[ ]:

#pca = PCA(n_components = 64)


# In[ ]:

#pca.fit(X_train)


# In[ ]:

#x_train = pca.fit_transform(X_train)


# In[ ]:

#x_test = pca.fit_transform(X_test)


# In[ ]:

from keras.optimizers import Adam


# In[ ]:

myAdam = Adam(lr = 0.0005)


# In[ ]:

input_dim = X_train.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
encoder = Dense(int(encoding_dim / 4), activation="relu")(encoder)
#encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
#decoder = Dense(int(encoding_dim / 4), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)


# In[ ]:

nb_epoch = 100
batch_size = 1024

autoencoder.compile(optimizer=myAdam, 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


# In[ ]:

fig = plt.figure(figsize=(15,9))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Autoencoder model loss',fontsize=36)
plt.ylabel('loss',fontsize=30)
plt.xlabel('epoch',fontsize=30)
plt.legend(['train', 'test'], loc='upper right',fontsize=30);
fig.savefig('model_loss.png', dpi=100)


# In[ ]:

all_data_encode = data[all_variables]
all_data_encode = all_data_encode.drop(['misstated'],axis=1)
#all_data_encode = all_data_encode.values.reshape(all_data_encode.shape[0], 1, all_data_encode.shape[1])

all_predictions_encode = autoencoder.predict(all_data_encode)

#all_test_encode = data[shifted_vars]
#all_test = all_test.drop(['misstated'],axis=1)


all_mse_encode = np.mean(np.power(all_data_encode - all_predictions_encode, 2), axis=1)
all_encode_error_df = pd.DataFrame({'lstm_ratio_reconstruction_error': all_mse_encode,'true_class': data['misstated'],
                                 'fyear':data['fyear'],'gvkey':data['gvkey']})


# In[ ]:

all_encode_error_df.shape


# In[ ]:

predictions = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})

error_df.describe()


# In[ ]:

error_df['true_class'].value_counts()


# In[ ]:

lower_threshold = 0.02#1000000
#higher_threshold = 10


# In[ ]:

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
fig.set_size_inches(15,9)
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
    ax.set_ylim(0,1)
#ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend(fontsize = 36)
plt.title("Reconstruction error for different classes",fontsize = 36)
plt.ylabel("Reconstruction error", fontsize = 30)
plt.xlabel("Data point index", fontsize = 30)
plt.show();
fig.savefig('reconstruction_error.png', dpi=100)


# In[ ]:

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[ ]:

fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:

import seaborn as sns
y_pred = [1 if ((e > lower_threshold) ) else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
classification_report(error_df.true_class, y_pred)


# In[ ]:




# In[ ]:

counts = data.groupby(['gvkey', 'misstated']).size().reset_index(name='counts')


# In[ ]:

#counts[counts['misstated']==True].sort_values(by=['counts'],ascending=False)


# In[ ]:

data[data['gvkey']==15354]['misstated']


# In[ ]:

data.shape


# In[ ]:

for i in all_variables:
    print (i)
    if i!='misstated':
        data[i+'_pre'] = data.groupby(['gvkey'])[i].shift(-1)


# In[ ]:

#data.to_csv("final_data_shifted.csv",index=False)


# In[ ]:

shifted_vars = []
for i in all_variables:
    if i!='misstated':
        shifted_vars.append(i+'_pre')


# In[ ]:

#data[shifted_vars].to_csv("final_data_shifted.csv",index=False)


# In[ ]:

data = data.dropna(how='any')


# In[ ]:

shifted_vars.append('misstated')


# In[ ]:

lstm_X_train, lstm_X_test = train_test_split(data[all_variables], test_size=0.2, random_state=123)
lstm_Y_train, lstm_Y_test = train_test_split(data[shifted_vars], test_size=0.2, random_state=123)
print(len(lstm_X_train))
lstm_X_train = lstm_X_train[lstm_X_train['misstated'] == False]
lstm_Y_train = lstm_Y_train[lstm_Y_train['misstated'] == False]

lstm_X_train = lstm_X_train.drop(['misstated'], axis=1)
lstm_Y_train = lstm_Y_train.drop(['misstated'], axis=1)

print(len(lstm_X_train))

lstm_y_test =lstm_X_test['misstated']
lstm_X_test = lstm_X_test.drop(['misstated'], axis=1)
lstm_Y_test = lstm_Y_test.drop(['misstated'], axis=1)

lstm_X_train = lstm_X_train.values
lstm_X_test = lstm_X_test.values

lstm_X_train.shape


# In[ ]:

lstm_X_test.shape


# In[ ]:

lstm_Y_test.shape


# In[ ]:

lstm_X_train = lstm_X_train.reshape((lstm_X_train.shape[0], 1, lstm_X_train.shape[1]))
lstm_X_test = lstm_X_test.reshape((lstm_X_test.shape[0], 1, lstm_X_test.shape[1]))


# In[ ]:

from keras import Sequential


# In[ ]:

from keras.layers import LSTM


# In[ ]:

model = Sequential()
model.add(LSTM(50, input_shape=(lstm_X_train.shape[1], lstm_X_train.shape[2])))
model.add(Dense(245))
model.compile(loss='mae', optimizer='adam')
# fit network
lstm_history = model.fit(lstm_X_train, lstm_Y_train, epochs=50, batch_size=1024, validation_data=(lstm_X_test, lstm_Y_test), verbose=2, shuffle=False)
# plot history


# In[ ]:

plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[ ]:

all_data = data[all_variables]
all_data = all_data.drop(['misstated'],axis=1)
all_data = all_data.values.reshape(all_data.shape[0], 1, all_data.shape[1])

all_predictions = model.predict(all_data)

all_test = data[shifted_vars]
all_test = all_test.drop(['misstated'],axis=1)


all_mse = np.mean(np.power(all_test - all_predictions, 2), axis=1)
all_lstm_error_df = pd.DataFrame({'lstm_ratio_reconstruction_error': all_mse,'true_class': data['misstated'],
                                 'fyear':data['fyear'],'gvkey':data['gvkey']})


# In[ ]:

all_lstm_error_df.head()


# In[ ]:

predictions = model.predict(lstm_X_test)



# In[ ]:

lstm_X_test = lstm_X_test.reshape((lstm_X_test.shape[0], lstm_X_test.shape[2]))
mse = np.mean(np.power(lstm_X_test - predictions, 2), axis=1)


# In[ ]:

lstm_error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': lstm_y_test})


# In[ ]:

lstm_error_df.describe()


# In[ ]:

lstm_groups = lstm_error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in lstm_groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=2.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
    ax.set_ylim(0,0.5)
#ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


# In[ ]:

lower_threshold = 0.00002#1000000


# In[ ]:

import seaborn as sns
lstm_y_pred = [1 if ((e > lower_threshold) ) else 0 for e in lstm_error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(lstm_error_df.true_class, lstm_y_pred)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
classification_report(lstm_error_df.true_class, lstm_y_pred)


# In[ ]:

count = 0
total_pred = []
for i in range(52270):
    if lstm_y_pred[i]==1 and y_pred[i]==1:
        count+=1
        total_pred.append(1)
    else:
        total_pred.append(0)
print (count)


# In[ ]:

sum(total_pred)


# In[ ]:

classification_report(lstm_error_df.true_class, total_pred)


# In[ ]:

conf_matrix = confusion_matrix(lstm_error_df.true_class, total_pred)
LABELS = ["Normal", "Fraud"]
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:

gvkey_train, gvkey_test = train_test_split(data['gvkey'], test_size=0.2, random_state=123)


# In[ ]:

fyear_train, fyear_test = train_test_split(data['fyear'], test_size=0.2, random_state=123)


# In[ ]:

combined_res = pd.DataFrame(
    {'lstm_pred': lstm_y_pred,
     'encoder_pred': y_pred,
     'lstm_error': lstm_error_df.reconstruction_error,
     'encoder_error': error_df.reconstruction_error,
     'truth': lstm_error_df.true_class,
     'gvkey':gvkey_test,
     'fyear':fyear_test
     
    })


# In[ ]:

combined_res.to_csv("combined_res.csv",index=False)


# In[ ]:




# In[ ]:

combined_res[combined_res['truth']==True]


# In[ ]:

data[((data['gvkey']==5021) & (data['fyear']==1987))]['misstated']


# In[ ]:




# In[ ]:

all_combined_res = pd.DataFrame(
    {
     'lstm_error_ratio': all_lstm_error_df['lstm_ratio_reconstruction_error'],
     'encoder_error_ratio': all_encode_error_df['lstm_ratio_reconstruction_error'],
     'truth': data['misstated'],
     'gvkey':data['gvkey'],
     'fyear':data['fyear']
     
    })


# In[ ]:

all_combined_res.describe()


# In[ ]:

all_combined_res.to_csv("all_combined_res.csv",index=False)


# In[ ]:

all_combined_res.head()


# In[ ]:



