# Importing model classes from the file model_classes.py
import model_classes as mc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import functools as f
import scipy
import lightgbm as lg

#garbage collection- Memory management
import gc

#Place the python file to the folder containing data

orders_prior,orders,orders_train,products,aisles,departments=mc.loadingdata()

#Checking for NaN's present in the orders data
orders.isnull().sum()

orders.loc[:,'days_since_prior_order']=orders.days_since_prior_order.fillna(0)

#merging prior orders with orders table to get the user_ids

prior_product_order_users=pd.merge(orders_prior,orders,on='order_id',how='inner')

#Assembling User properties to form features
#extracting unique users
user_prop=pd.DataFrame(orders.user_id.unique())
user_prop.columns=['user_id']
#User total order count- ranges from 4 to 100
user_total_orders=pd.DataFrame(orders[orders.eval_set=='prior'].groupby(['user_id'])['order_id'].nunique())
user_total_orders=user_total_orders.reset_index()
#Average number of days lapsed between two orders of a user
user_avg_days_per_ord=pd.DataFrame(orders[orders.eval_set=='prior'].groupby(['user_id'])['days_since_prior_order'].mean())
user_avg_days_per_ord=user_avg_days_per_ord.reset_index()
#Total products purchased by the user
user_tot_prods=pd.DataFrame(prior_product_order_users.groupby(['user_id'])['product_id'].count())
user_tot_prods=user_tot_prods.reset_index()
#Total unique products purchased by the user
user_distinct_prods=pd.DataFrame(prior_product_order_users.groupby(['user_id'])['product_id'].nunique())
user_distinct_prods=user_distinct_prods.reset_index()

merge_user_dfs=[user_prop,user_total_orders,user_avg_days_per_ord,user_tot_prods,user_distinct_prods]

#bringing all the user features into one table
user_property= f.reduce(lambda left,right: pd.merge(left,right,on='user_id'), merge_user_dfs)

user_property.columns=['user_id','user_total_orders','user_avg_days_per_ord','user_tot_prods','user_distinct_prods']
user_property['user_avg_basket']=user_property.user_tot_prods/user_property.user_total_orders
user_property.head()
#merging train and test orders with users
temp=orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order','order_hour_of_day']]
user_property=user_property.merge(temp,how='inner')
user_property['user_lst_ord_days_ratio'] = user_property.days_since_prior_order / user_property.user_avg_days_per_ord
user_property.user_lst_ord_days_ratio[user_property.user_lst_ord_days_ratio==float('inf')]=5000

#deleting unwanted datasets and collecting grabage
del user_prop,user_total_orders,user_avg_days_per_ord,user_tot_prods,user_distinct_prods,temp,merge_user_dfs
gc.collect()

#Assembling product features
#Extracting unique products
prod_prop=pd.DataFrame(prior_product_order_users.product_id.unique())
prod_prop.columns=['product_id']
#Total orders made till date for each product
prod_order_counts=pd.DataFrame(orders_prior.groupby('product_id')['order_id'].count())
#Total reorders for each product
prod_user_reorders=pd.DataFrame(orders_prior.groupby(['product_id'])['reordered'].sum())

prod_order_counts=prod_order_counts.reset_index()
prod_user_reorders=prod_user_reorders.reset_index()
merge_dfs=[prod_prop,prod_order_counts,prod_user_reorders]
#Bringing all product features into one table
prd_property=f.reduce(lambda left,right: pd.merge(left,right,on='product_id'), merge_dfs)
prd_property.columns=['product_id','prod_order_counts','prod_user_reorders']
prd_property['prod_reorder_ratio'] = pd.DataFrame(prd_property.prod_user_reorders / prd_property.prod_order_counts)

del prod_prop,prod_order_counts,prod_user_reorders,merge_dfs
gc.collect()

prd_property.head()
#Assembling all User product combination features
#Extracting unique user,product combination based on their prior orders
user_prod_prop=prior_product_order_users[['user_id','product_id']]
user_prod_prop.drop_duplicates(inplace=True)

#Getting the count of each user,product combination
user_prod_ord_cnt=pd.DataFrame(prior_product_order_users.groupby(['user_id','product_id'])['order_number'].count())
user_prod_ord_cnt=user_prod_ord_cnt.reset_index()

#Last order number where the user order the particular product
user_prod_last_ord=pd.DataFrame(prior_product_order_users.groupby(['user_id','product_id'])['order_number'].max())
user_prod_last_ord=user_prod_last_ord.reset_index()

#Average position of product in user's cart over his past orders
user_prd_avg_cart_pos=pd.DataFrame(prior_product_order_users.groupby(['user_id','product_id'])['add_to_cart_order'].mean())
user_prd_avg_cart_pos=user_prd_avg_cart_pos.reset_index()

merge_user_prod_dfs=[user_prod_prop,user_prod_ord_cnt,user_prod_last_ord,user_prd_avg_cart_pos]

#Bringing together user-product features
user_prd_property=f.reduce(lambda left,right: pd.merge(left,right,on=['user_id','product_id']), merge_user_prod_dfs)

user_prd_property.columns=['user_id','product_id','user_prod_ord_cnt','user_prod_last_ord','user_prd_avg_cart_pos']

usr_prd_ord_num = user_prod_last_ord.merge(prior_product_order_users[['user_id','product_id','order_id','order_number']],on                                           =['user_id','product_id','order_number'],how='left')

user_prd_property =  user_prd_property.merge(usr_prd_ord_num[['user_id','product_id','order_id']],on=['user_id','product_id'],                                                          how='left')

user_prd_property=user_prd_property.rename(columns={'order_id':'last_order_id'})
user_prd_property.head()

del user_prod_prop,user_prod_ord_cnt,user_prod_last_ord,user_prd_avg_cart_pos,merge_user_prod_dfs,usr_prd_ord_num
gc.collect()

#Merging User,Product and User-Product features
tot_merged_data=user_prd_property.merge(prd_property,how='inner',on='product_id').merge(user_property,how='inner',on='user_id')

#User-product ordering rate
tot_merged_data['user_prod_ord_rate']=tot_merged_data.user_prod_ord_cnt/tot_merged_data.user_total_orders

#Number of orders made by user since he last purchased the product
tot_merged_data['user_prod_ord_since_last_ord'] = tot_merged_data.user_total_orders - tot_merged_data.user_prod_last_ord

#Bringing in usr_id to train orders table
orders_train = orders_train.merge(orders[['order_id', 'user_id']], how='left', on='order_id')

tot_merged_data = tot_merged_data.merge(orders_train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

del user_prd_property,prd_property,user_property,
gc.collect()

del prior_product_order_users,orders_prior
gc.collect()

#Bringing in aisle and department to the final merged table with all features 
tot_merged_data=tot_merged_data.merge(products[['product_id','aisle_id','department_id']], on=['product_id'], how='left')
#tot_merged_data=tot_merged_data.merge(orders[orders.eval_set=='train'][['order_id','order_hour_of_day']],on=['order_id'],how='left')

uniq_ord=pd.DataFrame(tot_merged_data['last_order_id'].unique())
uniq_ord.columns=['last_order_id']

last_order_id_hr=pd.merge(uniq_ord, orders[['order_id','order_hour_of_day']],                           how='left', left_on=['last_order_id'], right_on=['order_id'])
del uniq_ord

last_order_id_hr=last_order_id_hr.rename(columns={'order_hour_of_day':'last_order_hour_of_day'})
last_order_id_hr=last_order_id_hr[['last_order_id','last_order_hour_of_day']]

tot_merged_data=tot_merged_data.merge(last_order_id_hr,on='last_order_id',how='left')

#Time difference in the hour of order between the last order and the latest order
tot_merged_data['time_diff_hr_last_ord']=abs(tot_merged_data.order_hour_of_day - tot_merged_data.last_order_hour_of_day).map                                            (lambda x: min(x, 24-x)).astype(np.int8)
#Fill reordered and user_lst_ord_days_ratio columns with 0 where NaN is present
tot_merged_data.loc[:, 'reordered'] = tot_merged_data.reordered.fillna(0)

tot_merged_data.loc[:, 'user_lst_ord_days_ratio'] = tot_merged_data.user_lst_ord_days_ratio.fillna(0)

#Finalized features to be used in modeling
features_use=['user_prod_ord_cnt','user_prd_avg_cart_pos','prod_order_counts','prod_user_reorders','prod_reorder_ratio',              'user_total_orders','user_avg_days_per_ord','user_tot_prods','user_distinct_prods','user_avg_basket',              'days_since_prior_order','order_hour_of_day','user_lst_ord_days_ratio','user_prod_ord_rate',              'user_prod_ord_since_last_ord','aisle_id','department_id','time_diff_hr_last_ord']

#Set training and test data
fin_train=tot_merged_data.loc[tot_merged_data.eval_set == "train",:]
#fin_train=fin_train[0:(len(fin_train)/100)]
df_test=tot_merged_data.loc[tot_merged_data.eval_set == "test",:]

#LightGBM
#lightgbm is a subclass of model_analysis class.
instagbm=mc.lightgbm(fin_train,'reordered')

#provides recommedation on modeling
instagbm.get_mod_recommendation()
instagbm.data_summary()

train_prop=0.9
eval_threshold=0.22
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}

#Model Evaluation using the fin_train data provided. Train-test split amount, Threshold for probability 
#and features to be used are given as inputs
instagbm.model_eval(train_prop,eval_threshold,params,features_use)

#Generates an ROC curve from the model evaluation function run above
instagbm.roc_curve_gen()
#generates and writes the output for test data. This has order_id and the products likely to be ordered for that order
instagbm.gen_test_output(df_test,0.22)

#Logistic Regression
#Classifier is a subclass of model_analysis class
instaLR=mc.classifier(fin_train,'reordered',LogisticRegression())

train_prop=0.9
eval_threshold=0.22

instaLR.model_eval(train_prop,eval_threshold,features_use)
instaLR.roc_curve_gen()
instaLR.gen_test_output(df_test,0.22)


#Linear Discriminant Analysis
instaLDA=mc.classifier(fin_train,'reordered',LinearDiscriminantAnalysis())
train_prop=0.9
eval_threshold=0.22
instaLDA.model_eval(train_prop,eval_threshold,features_use)
instaLDA.roc_curve_gen()
#instaLDA.gen_test_output(df_test,0.22)

#KNN CLassifier
instaKNN=mc.classifier(fin_train,'reordered',KNeighborsClassifier())
train_prop=0.9
eval_threshold=0.22
instaKNN.model_eval(train_prop,eval_threshold,features_use)
instaKNN.roc_curve_gen()
#instaKNN.gen_test_output(df_test,0.22)

#DecisionTreeClassifier
instaDTC=mc.classifier(fin_train,'reordered',DecisionTreeClassifier())
train_prop=0.9
eval_threshold=0.22
instaDTC.model_eval(train_prop,eval_threshold,features_use)
instaDTC.roc_curve_gen()
#instaDTC.gen_test_output(df_test,0.22)

#GaussianNB
instaGNB=mc.classifier(fin_train,'reordered',GaussianNB())
train_prop=0.9
eval_threshold=0.22
instaGNB.model_eval(train_prop,eval_threshold,features_use)
instaGNB.roc_curve_gen()
#instaGNB.gen_test_output(df_test,0.22)
