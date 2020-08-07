#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st


# In[172]:


os.getcwd()


# In[173]:


#os.chdir()


# In[174]:


#%pip install plotnine 

st.title("""#Data Science project to check a person is defaulter or not defaulter""")
st.write("""DataFrame""")


# In[175]:


from plotnine import *


# In[176]:


import missingno as msno


# In[177]:


r='C:/Users/ADITEYA RAJ/Desktop/Excelr project/bank_final.csv'
df=pd.read_csv(r)
df


# In[178]:


st.subheader('Data Information')
st.dataframe(df)


# In[179]:


df.describe()


# In[180]:


msno.heatmap(df)


# In[181]:


msno.bar(df)


# In[182]:


df.info()


# In[183]:


df.dtypes


# In[184]:


df.isna().sum()


# In[185]:


df.nunique()


# #COLUMNS WIZE ANALYSIS OF DATA

# ##1.]NAME-:

# Name are the name of the customers/shops/factory who took loan from a particular Bank given in the banks columns

# In[186]:


len(df.Name.unique())


# out of which 140884 are unique Names

# In[187]:


df.Name.isna().sum()


# 8 columns are having no name so we can do any analysis on that or a can do any imputation on those 8 values

# #droping "Name" Column because of  140884 are unique Names

# In[188]:


drop=["Name"]
df.drop(drop,inplace=True,axis=1)


# #2.]City-:

# In[189]:


df.City.isna().sum()# only 1 nan value in the city columns


# In[190]:


df.drop(df[df.City.isna()].index,inplace=True)


# In[191]:


df.City.value_counts()


# In[192]:


m=dict(df.City.value_counts())


# In[193]:


name=[name for name, num in m.items() if num >600] 

num=[num for name, num in m.items() if num >600]   


# In[194]:


len(df.City.unique())


# In[195]:


name,num


# ['LOS ANGELES','NEW YORK','MIAMI','CHICAGO','HOUSTON','BROOKLYN','DALLAS','SAN DIEGO','PHOENIX','PHILADELPHIA','LAS VEGAS','ATLANTA','ROCHESTER'] are the City which have customers more than 600 who took a loan

# #3.]State

# In[196]:


df.State.isna().sum()#so we have 2 nan value in State column so droping these nan as imputation is good for these 2 values


# In[197]:


df.drop(df[df.State.isna()].index,inplace=True)


# In[198]:


df.shape


# In[199]:


df.State.unique()# are the unique States present in the dataset


# In[200]:


n=dict(df.State.value_counts())


# In[201]:


#plt.figure(figsize=(30,10))
#plt.bar(n.keys(),n.values(),width=0.3)


# In[202]:


#plt.figure(figsize=(30,10))
#ggplot(df)+aes('State',fill='MIS_Status')+geom_histogram()+ theme(axis_text_x = element_text(size = 5))


# #so we see from the plot has State can be a good variable for prediction of CHGOFF and PIF

# #4.]ZIP

# In[203]:


df.Zip.isna().sum()


# In[204]:


df.Zip.value_counts()


# In[205]:


#ggplot(df)+aes("Zip",fill="MIS_Status")+geom_bar(width=4)


# In[206]:


df.Zip.unique()
  #so all address of customers are from different zip code


# #5.]Bank

# In[207]:


df.Bank.isna().sum()# we have 147 nan values in the Bank column so we need to drop these as imputation will not be good for these nominal data


# In[208]:


df.drop(df[df.Bank.isna()].index,inplace=True)


# In[209]:


df.Bank.unique()


# In[210]:


len(df.Bank.unique())


# so we have 2933 value which are unique in our Bank columns and rest are the repition of these Banks 

# In[211]:


#ggplot(df)+aes('Bank',fill='MIS_Status')+geom_histogram()+coord_flip()+ theme(axis_text_x = element_text(size = 5))


# In[212]:


df.shape


# #6.]BankState

# In[213]:


df.BankState.isna().sum()


# In[214]:


df.drop(df[df.BankState.isna()].index,inplace=True)


# In[215]:


df.BankState.unique()


# In[216]:


#ggplot(df)+aes('BankState',fill='MIS_Status')+geom_histogram()+coord_flip()+ theme(axis_text_x = element_text(size = 5))


# so we see that NC AND IL have the most no of fraud cases so BankState is a good variable for prediction
# 
# 
# ```
# 
# 

# In[217]:


l=list(df.BankState.iloc[:]==df.State.iloc[:])
k=list(df.MIS_Status.iloc[:])
t,s=0,0
m=[]
for j in k:
  if j=='CHGOFF':
    m.append(False)
  else:
    m.append(True)


for i in l:
  if i ==True:
    t=t+1
  else:
    s=s+1
t,s


# In[218]:


no=0
for i in range(len(df)):
  if l[i]==False and m[i]==False:
    no=no+1
no


# In[219]:


nu=0
for i in range(len(df)):
  if l[i]==True and m[i]==False:
    nu=nu+1
nu


# In[220]:


no/s*100,nu/t*100


# #so from this analysis we get to know that 34% of people who come under CHGOFF are customer who belong to aparticular state and they take a loan from the bank from different BankState.  

# #7.]CCSC

# In[221]:


df.CCSC.isna().sum()


# In[222]:


df.CCSC.value_counts()


# In[223]:


#d=dict(df.CCSC.value_counts())
#d.keys(),d.values()


# In[224]:


#ggplot(df)+aes('CCSC',fill='MIS_Status')+geom_histogram()


# ccsc will be a good variable for prediction

# #8.]ApprovalDate

# In[225]:


df.ApprovalDate.isna().sum()


# In[226]:


len(df.ApprovalDate.value_counts())


# In[227]:


#ggplot(df)+aes('ApprovalDate',fill='MIS_Status')+geom_histogram()


# In[228]:


df['ApprovalDate'].value_counts()


# #9.]Date columns-['ApprovalDate','DisbursementDate', 'ChgOffDate']

# In[229]:


# converting Date text columns to datetime object
date_cols = ['ApprovalDate','DisbursementDate', 'ChgOffDate']
for dates in date_cols:
    df[dates] = pd.to_datetime(df[dates])


# In[230]:


df.ChgOffDate.isna().sum()


# In[231]:


df.shape


# In[232]:


109533/149999# 74% of data in ChgoffDate is filled with nan val value so drop this column -ChgOffDate


# In[233]:


# droping the ChgOffDate Column
dro=['ChgOffDate']
df.drop(dro,inplace=True,axis=1)


# In[234]:


df.DisbursementDate.isna().sum()


# In[235]:


df.drop(df[df.DisbursementDate.isna()].index,inplace=True)


# In[236]:


df.shape


# #10.]ApprovalFY

# In[237]:


df.ApprovalFY.isna().sum()


# In[238]:


df.ApprovalFY.value_counts()


# In[239]:


#ggplot(df)+aes('ApprovalFY',fill='MIS_Status')+geom_bar(width=2)# this can be a good variable for prediction of the target variable


# In[240]:


df.columns


# #11.]Term

# In[241]:


df.Term.isna().sum()


# In[242]:


df.Term.value_counts()


# In[243]:


#ggplot(df)+aes(('Term'),fill='MIS_Status')+geom_bar(width=20)


# In[244]:


df.columns


# #12.] NoEmp

# In[245]:


df.NoEmp.isna().sum()


# In[246]:


df.NoEmp.value_counts()


# #13.]NewExist

# In[247]:


df.NewExist.isna().sum()


# In[248]:


df.NewExist.value_counts()


# #14.]CreateJob

# In[249]:


df.CreateJob.value_counts()


# In[ ]:





# #RetainedJob

# In[250]:


df.RetainedJob.isna().sum()


# FranchiseCode

# In[251]:


df.UrbanRural.isna().sum()


# In[252]:


df.UrbanRural.value_counts()


# UrbanRural

# In[253]:


df.UrbanRural.isna().sum()


# In[254]:


df.UrbanRural.value_counts()


# #RevLineCr

# In[255]:


df.RevLineCr.isna().sum()


# In[256]:


df.drop(df[df.RevLineCr.isna()].index,inplace=True)


# In[257]:


df.RevLineCr.value_counts()


# In[258]:


df.drop(df[df['RevLineCr']=='0'].index,inplace=True)
df.drop(df[df['RevLineCr']=='`'].index,inplace=True)
df.drop(df[df['RevLineCr']=='1'].index,inplace=True)
df.drop(df[df['RevLineCr']==','].index,inplace=True)
df.drop(df[df['RevLineCr']=='T'].index,inplace=True)


# In[259]:


df.shape


# RevLineCr column have garbage values 
# and 23 nan values

# In[260]:


df.LowDoc.isna().sum()


# In[261]:


df.LowDoc.value_counts()


# In[262]:


df.drop(df[df['LowDoc']=='C'].index,inplace=True)
df.drop(df[df['LowDoc']=='1'].index,inplace=True)


# LowDoc column have 84 garbage values

# In[263]:


df.shape


# In[264]:


#term ,DisbursementGross,df.Charge offgross,sbapproval,


# #DisbursementGross

# In[265]:


df.DisbursementGross.isna().sum()


# In[266]:


df.DisbursementGross.value_counts()


# remving $ from the values in each column

# In[267]:


#stripping $ and , sign from currency columns and converting into float64
currency_cols = ['DisbursementGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv','BalanceGross']
for cols in currency_cols:
   df[cols] = df[cols].str.replace('$', '').str.replace(',', '')


# In[268]:


df


# In[269]:


df.DisbursementGross.isna().sum()


# In[270]:


df.ChgOffPrinGr.isna().sum()


# In[271]:


df.GrAppv.isna().sum()


# In[272]:


df.BalanceGross.isna().sum()


# In[273]:


df.isna().sum()


# #MIS_Status

# In[274]:


df.MIS_Status.value_counts()
#so thid has 2 categorie so this is the best variable for target variable 


# In[275]:


df.MIS_Status.isna().sum()


# In[276]:


df.drop(df[df.MIS_Status.isna()].index,inplace=True)


# In[277]:


df.isna().sum()


# In[278]:


df.shape


# In[279]:


df.info()


# In[280]:


df['State'].dtype


# In[281]:


cat_features=[i for i in df.columns if df.dtypes[i]=='object']
len(cat_features),cat_features


# In[282]:


df


# In[283]:


d=['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']
df.DisbursementGross=df.DisbursementGross.astype('float')
df.BalanceGross=df.BalanceGross.astype('float')
df.ChgOffPrinGr=df.ChgOffPrinGr.astype('float')
df.GrAppv=df.GrAppv.astype('float')
df.SBA_Appv=df.SBA_Appv.astype('float')


# In[284]:


df.info()


# In[285]:


numeric_features=[i for i in df.columns if df.dtypes[i]!='object']
len(numeric_features),numeric_features


# #CORRILATION

# In[286]:


from scipy.stats import pearsonr


# In[287]:


corr=df.corr(method='pearson')
corr


# In[288]:


corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# #ApprovalFY and UrbanRural 0.72
# #Term and SBA_Appv 0.54
# #ApprovalFY and CCSC 0.59
# #UrbanRural	and CCSC 0.49
# #DisbursementGross AND SBA_Appv 0.89
# #DisbursementGross AND GrAppv 0.93
# #SBA_Appv AND GrAppv  0.97
# 
# 

# In[289]:


drp=['DisbursementGross','GrAppv','UrbanRural']
df.drop(drp,inplace=True,axis=1)


# In[290]:


dr=['ApprovalFY']
df.drop(dr,inplace=True,axis=1)
df.columns


# In[291]:


df.shape


# In[292]:


corri=df.corr(method='pearson')


# In[293]:


corri.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[294]:


Newdata=df
data=df


# In[295]:


df


# In[296]:


cat_features=[i for i in df.columns if df.dtypes[i]=='object']
len(cat_features),cat_features


# In[297]:


df.City.value_counts()


# In[298]:


data.Zip.value_counts()


# In[299]:


dr=['City','Zip','State','Bank','BankState','ApprovalDate','DisbursementDate']
data.drop(dr,inplace=True,axis=1)


# In[300]:


data.head()


# In[301]:


#c=data.ApprovalDate.value_counts().to_dict()


# In[302]:


#data.ApprovalDate=data.ApprovalDate.map(c)
#data.head()


# In[303]:


#di=data.DisbursementDate.value_counts().to_dict()


# In[304]:


#data.DisbursementDate=data.DisbursementDate.map(di)
#data.head()


# In[305]:


#df_frequency_map = data.State.value_counts().to_dict()


# In[306]:


#data.State = data.State.map(df_frequency_map)



data.head()


# In[307]:


#f=data.Bank.value_counts().to_dict()


# In[308]:


#data.Bank = data.Bank.map(f)

#data.head()


# In[309]:


#g=data.BankState.value_counts().to_dict()


# In[310]:


#data.BankState=data.BankState.map(g)
#data.head()


# In[311]:


df


# In[312]:


#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transformer
#column_trans=make_column_transformer((OneHotEncoder(),['RevLineCr','LowDoc']),remainder='passthrough')
#df = (column_trans.fit_transform(df))


# In[313]:


from sklearn import preprocessing 
  


# In[314]:


# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['MIS_Status']= label_encoder.fit_transform(df['MIS_Status']) 


# In[315]:


df['RevLineCr']= label_encoder.fit_transform(df['RevLineCr']) 


# In[316]:


df['LowDoc']= label_encoder.fit_transform(df['LowDoc']) 


# In[317]:


df.info()


# In[318]:


#creating independent and dependent features
columns1=df.columns.tolist()
columns_X=[c for c in columns1 if c not in ['MIS_Status']]
target='MIS_Status'
target


# In[319]:


len(columns_X),columns_X


# In[320]:


df


# Independent Features
# ['State',
#   'Bank',
#   'BankState',
#   'CCSC',
#   'ApprovalDate',
#   'Term',
#   'NoEmp',
#   'NewExist',
#   'CreateJob',
#   'RetainedJob',
#   'FranchiseCode',
#   'RevLineCr',
#   'LowDoc',
#   'DisbursementDate',
#   'BalanceGross',
#   'ChgOffPrinGr',
#   'SBA_Appv'])

# In[321]:


state=np.random.RandomState(42)
x=df[columns_X]
y=df[target]
max(x.SBA_Appv),min(x.SBA_Appv)


# In[322]:


#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
Model_ETC=ExtraTreesClassifier()


# In[323]:


Model_ETC.fit(x,y)

plt.bar(range(len(Model_ETC.feature_importances_)),Model_ETC.feature_importances_)


# In[324]:


print(Model_ETC.feature_importances_)


# In[325]:


#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
Model_ETC=ExtraTreesClassifier(n_estimators=10)
Model_ETC.fit(x,y)
print(Model_ETC.feature_importances_)
plt.bar(range(len(Model_ETC.feature_importances_)),Model_ETC.feature_importances_)


# In[326]:


#len(x.State.unique())
#x.nunique()


# In[327]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# In[328]:


# The classes are heavily skewed we need to solve this issue.
print('No Frauds', round(df['MIS_Status'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['MIS_Status'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


# In[329]:


import seaborn as sns
print('Distribution of the Classes in the subsample dataset')
print(df['MIS_Status'].value_counts()/len(df))



sns.countplot('MIS_Status', data=df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# #OVERSAMPLING

# In[330]:


from imblearn.over_sampling import SMOTE
smote=SMOTE( random_state=42)


# In[331]:


x_train_smote,y_train_smote=smote.fit_sample(x_train,y_train)


# In[332]:


from collections import Counter
print("before SMOTE:",Counter(y_train))
print('after SMOTE:',Counter(y_train_smote))


# In[333]:


print('Distribution of the Classes in the subsample dataset')
sns.countplot(y_train_smote, data=df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[334]:


from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[335]:


cv_method = KFold(n_splits=2, shuffle=True)


# In[336]:


#param_g=[{'max_depth':[1,2,3,],'learning_rate':[0.1,1],'gama':[0.0],'reg_lambda':[0.0,1.0]}]


# In[337]:


#optimal_param=GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',subsample=0.75,colsample_bytree=0.5),
                           #param_grid=param_g,scoring='roc_auc',n_jobs=10,cv=cv_method)


# In[338]:


#optimal_param.fit(x_train_smote,y_train_smote)


# In[339]:


#xgb2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           #colsample_bytree=1, max_depth=7)
#param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
#grid = GridSearchCV(estimator = xgb2,
                              #param_grid = param,
                               #scoring = 'neg_mean_squared_error',
                               #cv = cv_method,
                               #n_jobs = -1)


# In[340]:



xgb2 = xgb.XGBClassifier(n_estimators=100)
param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
grid = GridSearchCV(estimator = xgb2,
                               param_grid = param,
                               scoring = 'roc_auc',
                               cv = cv_method)


# In[ ]:


#xgb2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           #colsample_bytree=1, max_depth=7)
#param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
#grid = GridSearchCV(estimator = xgb2,
                               #param_grid = param_g,
                               #scoring = 'roc_auc',
                               #cv = cv_method,
                               #n_jobs = -1)


# In[342]:


grid.fit(x_train_smote,y_train_smote)


# In[343]:


print(grid.best_params_)


# In[344]:


y_pred=grid.predict(x_test)


# In[345]:


from sklearn.metrics import accuracy_score 
print("ACCURACY",accuracy_score(y_test,y_pred))
pd.crosstab(y_test,y_pred)


# In[346]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[347]:


#plot_confusion_matrix(grid,x_test,y_test,values_format='b')


# In[348]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = grid.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('oversampling model: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='roc')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[349]:


#UNDERSAMPLING


# In[350]:


from imblearn.under_sampling import NearMiss


# In[351]:


nm= NearMiss()


# In[352]:


x_res,y_res=nm.fit_sample(x_train,y_train)


# In[353]:


from collections import Counter
print("before SMOTE:",Counter(y_train))
print('after SMOTE:',Counter(y_res))


# In[354]:


print('Distribution of the Classes in the subsample dataset')
sns.countplot(y_res, data=df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[355]:


cv_method = KFold(n_splits=10, shuffle=True)


# In[356]:


xgb1 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
grid1 = GridSearchCV(estimator = xgb1,
                               param_grid = param,
                               scoring = 'neg_mean_squared_error',
                               cv = cv_method)


# In[357]:


grid1.fit(x_res,y_res)


# In[358]:


y_pred1=grid1.predict(x_test)


# In[359]:


from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test,y_pred1))
pd.crosstab(y_test,y_pred1)


# In[360]:


print(classification_report(y_test,y_pred1))


# In[361]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = grid1.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('undersampling model: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='roc')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[362]:



# ML Pkg
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[363]:


#import pickle
#saving model to disk
#pickle_out=open('model.pkl','wb')
#pickle.dump(grid1,pickle_out)


# In[364]:


#pickle_out1=open('model.pkl','wb')
#pickle.dump(grid,pickle_out1)


# In[365]:


#pickle_out.close()


# In[366]:


#pickle_out1.close()


# In[367]:


clf = DecisionTreeClassifier()


# In[368]:


clf.fit(x_train,y_train)


# In[369]:


# Model Accuracy Score
clf.score(x_test,y_test)


# In[370]:


y_pred_rf=grid.predict(x_test)
print(accuracy_score(y_test,y_pred_rf))
pd.crosstab(y_test,y_pred_rf)


# In[371]:


print(classification_report(y_test,y_pred_rf))


# In[372]:


from sklearn.neighbors import KNeighborsClassifier


# In[373]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[374]:



# Fit
knn.fit(x_train,y_train)


# In[375]:



# Model Accuracy Score
knn.score(x_test,y_test)


# In[377]:


y_pred_knn=grid.predict(x_test)
print(accuracy_score(y_test,y_pred_knn))
pd.crosstab(y_test,y_pred_knn)


# In[378]:


print(classification_report(y_test,y_pred_knn))


# In[393]:


random=RandomForestClassifier()


# In[394]:


random.fit(x_train,y_train)


# In[397]:


random.score(x_train,y_train)


# In[396]:


y_pred_r=random.predict(x_test)
pd.crosstab(y_test,y_pred_r)
print(accuracy_score(y_test,y_pred_r))


# In[401]:


print(classification_report(y_test,y_pred_r))


# In[407]:


x_test_m = x_test.values
x_train_m = x_train.values


# In[408]:


xg = xgb.XGBClassifier(n_estimators=50)
xg.fit(x_train_m,y_train)
xg.score(x_train_m,y_train)


# In[409]:


y_pred_xg=xg.predict(x_test_m)
pd.crosstab(y_test,y_pred_xg)
print(accuracy_score(y_test,y_pred_xg))


# In[410]:


print(classification_report(y_test,y_pred_xg))


# In[382]:


import joblib


# In[383]:


model_file_clf = open("decision_tree_model.pkl","wb")
joblib.dump(clf,model_file_clf)
model_file_clf.close()


# In[384]:


model_file_knn = open("knn_model.pkl","wb")
joblib.dump(knn,model_file_knn)
model_file_knn.close()


# In[412]:


model_r=open("rf_model.pkl","wb")
joblib.dump(random,model_r)
model_file_knn.close()


# In[411]:


model_xgb=open("xgb_model.pkl","wb")
joblib.dump(xg,model_xgb)
model_file_knn.close()


# In[400]:





# In[386]:


x.to_csv("x.csv")


# In[ ]:





# In[ ]:


#df1=pd.DataFrame(df)
#df1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




