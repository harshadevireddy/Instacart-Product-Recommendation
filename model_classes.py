
# coding: utf-8

# In[2]:

import numpy as np 
import pandas as pd # data loading and processing 
#Importing sklean for Classification modeling
from sklearn import model_selection as mod_sel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc,roc_auc_score
#Loading matplotlib for plotting
import matplotlib.pyplot as plt


# In[3]:

import lightgbm as lg



# In[4]:

def loadingdata():
    order_products__prior=pd.read_csv('order_products__prior.csv')
    order_users_all=pd.read_csv( 'orders.csv')
    order_products_train=pd.read_csv( 'order_products__train.csv')
    products=pd.read_csv( 'products.csv')
    aisles=pd.read_csv('aisles.csv')
    departments=pd.read_csv('departments.csv')
    print('Data Loading completed')
    return order_products__prior,order_users_all,order_products_train,products,aisles,departments


# In[8]:

class model_analysis:
    def __init__(self,data,target_col):
        self.data=data
        self.target_col=target_col
    def get_mod_recommendation(self):
        if not np.issubdtype(self.data[self.target_col].dtype, np.number) :
            print('Your target is not a numeric. So you might try a classification model')
        
        elif self.data[self.target_col].nunique()<(len(self.data.index)/10):
            print('Your target variable has very few categories. So a classification model is recommended.Use classifer or lightgbm\ subclasses')
        
        else:
            print('Your target variable has too many numeric values. Regression model is recommended')
    
    def data_summary(self):
        print('Your data has ',len(self.data.index),' records and ',len(self.data.columns),' columns.')
        print('\n-----Printing the summary of each column--------')
        print(self.data.describe(include='all'))
    
    def train_val_split(self,train_prop):
        self.train_prop=train_prop
        X_train, X_val, Y_train, Y_val = mod_sel.train_test_split(self.data.drop(self.target_col, axis=1), self.data[self.target_col],                                                    test_size=1-self.train_prop, random_state=42)
        print('Train set size: ',len(X_train.index),' records')
        print('Test set size: ',len(X_val.index),' records')
        return X_train, X_val, Y_train, Y_val
        

class classifier(model_analysis):
    def __init__(self,data,target_col,model_func):
        model_analysis.__init__(self,data,target_col)
        self.model_func=model_func
    def fit_data(self):
        self.X=self.data[self.data.columns.difference([self.target_col])]
        self.Y=self.data[self.target_col]
        self.model=self.model_func.fit(X,Y)
        print('Scoring Accuracy: ',self.model.score(self.X,self.Y))
    def model_eval(self,train_prop,eval_threshold,features_use):
        self.train_prop=train_prop
        self.eval_threshold=eval_threshold
        self.features_use=features_use
        self.X_train, self.X_val, self.Y_train, self.Y_val=mod_sel.train_test_split(self.data[self.features_use],\
                                                   self.data[self.target_col],test_size=1-self.train_prop, random_state=42)
        self.model= self.model_func.fit(self.X_train, self.Y_train)
        self.probs =self.model_func.predict_proba(self.X_val)
        self.predicted=(self.model_func.predict_proba(self.X_val) > self.eval_threshold).astype(int)[:,1]
        print(classification_report(self.Y_val, self.predicted))
        print(confusion_matrix(self.Y_val, self.predicted))
    def roc_curve_gen(self):
        self.fpr, self.tpr, _ = roc_curve(self.Y_val, self.probs[:, 1])
        self.roc_auc=auc(self.fpr,self.tpr)
        print ('Area under ROC curve: ',self.roc_auc)
        plt.figure()
        self.lw = 2
        plt.plot(self.fpr, self.tpr, color='darkorange',
                 lw=self.lw, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=self.lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    def gen_test_output(self,df_test,test_threshold):
        self.df_test=df_test
        self.test_threshold=test_threshold
        self.probs =self.model.predict_proba(self.df_test[self.features_use])
        print(len(self.probs))
        self.preds =( self.probs> self.test_threshold).astype(int)[:,1]

        self.df_test['pred'] = self.preds
        self.df_test.loc[:, 'product_id'] = self.df_test.product_id.astype(str)
        self.submission=self.df_test.groupby('order_id')['product_id'].agg(lambda x: ' '.join(set(x)))
        self.submission=pd.DataFrame(self.submission)
        self.submission=self.submission.reset_index()
        self.submission.columns=['order_id','products']
        self.submission.loc[:,'products']=self.submission.products.fillna('None')
        print('Product recommendations saved to output_classifier.csv file')
        self.submission.to_csv("output_classifier.csv", index=False)


# In[9]:

class lightgbm(model_analysis):
    def __init__(self,data,target_col):
        model_analysis.__init__(self,data,target_col)
    def model_eval(self,train_prop,eval_threshold,params,features_use):
        self.train_prop=train_prop
        self.eval_threshold=eval_threshold
        self.params=params
        self.features_use=features_use
        self.X_train, self.X_val, self.Y_train, self.Y_val=mod_sel.train_test_split(self.data[self.features_use],\
                                                   self.data[self.target_col],test_size=1-self.train_prop, random_state=42)
        self.d_train = lg.Dataset(self.X_train,label=self.Y_train,categorical_feature=['aisle_id', 'department_id'])
 
       
        self.ROUNDS = 100
        self.bst = lg.train(self.params, self.d_train, self.ROUNDS)
        self.probs =self.bst.predict(self.X_val)
        self.predicted=(self.bst.predict(self.X_val) > self.eval_threshold).astype(int)
        print(classification_report(self.Y_val, self.predicted))
        print(confusion_matrix(self.Y_val, self.predicted))
    def roc_curve_gen(self):
        self.fpr, self.tpr, _ = roc_curve(self.Y_val, self.probs)
        self.roc_auc=auc(self.fpr,self.tpr)
        print ('Area under ROC curve: ',self.roc_auc)
        plt.figure()
        self.lw = 2
        plt.plot(self.fpr, self.tpr, color='darkorange',
                 lw=self.lw, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=self.lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    def gen_test_output(self,df_test,test_threshold):
        self.df_test=df_test
        self.test_threshold=test_threshold
        self.preds = (self.bst.predict(self.df_test[self.features_use]) > self.test_threshold).astype(int)
        self.df_test['pred'] = self.preds
        self.df_test.loc[:, 'product_id'] = self.df_test.product_id.astype(str)
        self.submission=self.df_test.groupby('order_id')['product_id'].agg(lambda x: ' '.join(set(x)))
        self.submission=pd.DataFrame(self.submission)
        self.submission=self.submission.reset_index()
        self.submission.columns=['order_id','products']
        self.submission.loc[:,'products']=self.submission.products.fillna('None')
        print('Product recommendations saved to output_lgbm.csv file')
        self.submission.to_csv("output_lgbm.csv", index=False)

# In[ ]:



