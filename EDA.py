
# coding: utf-8

#import libraries and modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
plt.style.use('ggplot')

#load and read all six csv datasets as dataframe objects and then look into them

# aisles.csv -- contains aisles identifier and name of the aisles
# unique in aisle_id
aisles = pd.read_csv('.../aisles.csv')
print('Total aisles: {}'.format(aisles.shape))
print(aisles.head(5))

#cheking for missing values
missing = aisles.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

# departments.csv -- contains department identifer and name of the department
# unique in department_id
departments = pd.read_csv('../departments.csv')
print('Total departments: {}'.format(departments.shape))
print(departments.head(5))

missing = departments.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

# products.csv -- contains all product information
# unique in product_id
products = pd.read_csv('../products.csv')
print('Total products: {}'.format(products.shape))
print(products.head(5))

missing = products.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

# orders.csv -- contains all order and customer purchase information
# tells us which set (prior, train, test) an order belongs to
# unique in order_id
orders = pd.read_csv('../orders.csv')
print('Total orders: {}'.format(orders.shape))
print(orders.head(5))

missing = orders.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

# order_products_prior.csv -- contains previous orders for all customers
#unique in order_id & product_id
orders_prior = pd.read_csv('../order_products_prior.csv')
print('Total prior orders: {}'.format(orders_prior.shape))
print(orders_prior.head(5))

missing = orders_prior.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

# order_products_train.csv -- contains last orders for all customers
#unique in order_id & product_id
orders_train = pd.read_csv('../order_products_train.csv')
print('Total last orders: {}'.format(orders_train.shape))
print(orders_prior.head(5))

missing = orders_train.isnull().sum().sort_values(ascending=False)
print("Missing data: ", missing)

#combine order_products_prior.csv and order_products_train.csv into one dataframe
orders_products = pd.concat([orders_prior, orders_train], axis=0)
print("Total is: ", orders_products.shape)
print(orders_products.head(5))

# combine aisles.csv, departments.csv and products.csv into one dataframe
products_data = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')

# to make product_name and aisle more usable and remove spaces between words
products_data.product_name = products_data.product_name.str.replace(' ', '_').str.lower()
products_data.aisle = products_data.aisle.str.replace(' ', '_').str.lower()

print("Total rows: ", products_data.shape)
print(products_data.head(5))

# 'eval_set' in orders.csv informs the given order goes to which of the three datasets (prior, train or test)
count_set = orders.eval_set.value_counts()
print(count_set)

# define function to count the number of unique customers
def customer_count(x):
    return len(np.unique(x))

customer = orders.groupby("eval_set")["user_id"].aggregate(customer_count)
print(customer)

# number of unique orders and unique products
unique_orders = len(set(orders_products.order_id))
unique_products = len(set(orders_products.product_id))
print("There are %s orders for %s products" %(unique_orders, unique_products))

#number of products that customers usually order
data = orders_products.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
data = data.add_to_cart_order.value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(data.index, data.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.ylabel('Number of Orders', fontsize=12)
plt.xlabel('Number of products added in order', fontsize=12)
plt.show()

#the number of orders made by each costumer in the whole dataset
data = orders.groupby('user_id')['order_id'].apply(lambda x: len(x.unique())).reset_index()
data = data.groupby('order_id').aggregate("count")

sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(data.index, data.user_id)
plt.ylabel('Numbers of Customers')
plt.xlabel('Number of Orders per customer')
plt.xticks(rotation='vertical')
plt.show()

# time at which people usually order products
# hours of order in a day
data = orders.groupby("order_id")["order_hour_of_day"].aggregate("sum").reset_index()
data = data.order_hour_of_day.value_counts()
sns.set_style('darkgrid')
plt.figure(figsize=(12, 8))
sns.barplot(data.index, data.values)
plt.ylabel('Number of orders', fontsize=12)
plt.xlabel('Hours of order in a day', fontsize=12)
plt.title('When people buy groceries online?', fontweight='bold')
plt.xticks(rotation='vertical')
plt.show()

# order and day of a week
data = orders.groupby("order_id")["order_dow"].aggregate("sum").reset_index()
data = data.order_dow.value_counts()
days=[ 'sat','sun', 'mon', 'tue', 'wed', 'thu', 'fri']
plt.figure(figsize=(12,8))
sns.barplot(data.index, data.values)
plt.ylabel('Number of orders', fontsize=12)
plt.xlabel('Days of order in a week', fontsize=12)
plt.title("Frequency of order by week day", fontweight='bold')
plt.xticks(rotation='vertical')
plt.xticks(np.arange(7), ( 'sat','sun', 'mon', 'tue', 'wed', 'thu', 'fri'))
plt.show()

# When do they order again? time interval between orders
plt.figure(figsize=(12,8))
sns.countplot(x="days_since_prior_order", data=orders)
plt.ylabel('Nuumber of Orders', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.xticks(np.arange(31))
plt.title("Time interval between orders", fontsize=15)
plt.show()

# percentage of reorders in prior set #
print(orders_prior.reordered.sum() / orders_prior.shape[0])

# percentage of reorders in train set #
print(orders_train.reordered.sum() / orders_train.shape[0])

#reorder frequency
data = orders_products.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
data['Ratios'] = data["Total_products"].apply(lambda x: x /data['Total_products'].sum())
data
data = data.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(6, 8))
sns.barplot(data.index, data.values)
plt.ylabel('Number of Products', fontsize=12)
plt.xlabel('Not Reordered = 0 and Reordered = 1', fontsize=12)
plt.title('Reorder Frequency', fontweight='bold')
plt.ticklabel_format(style='plain', axis='y')
plt.show()


#merge products_data with the orders_products data.
orders_products_all = pd.merge(orders_products, products_data, on='product_id', how='left')
print("Total rows: ", orders_products_all.shape)
print(orders_products_all.head(5))

#merge products_data with the orders_prior data.
orders_prior_all = pd.merge(orders_prior, products_data, on='product_id', how='left')
print("Total rows: ", orders_prior_all.shape)
print(orders_prior_all.head(5))

#which products are ordered the most
data = orders_products.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'}).reset_index()
data = pd.merge(data, products[['product_id', 'product_name']], how='left', on=['product_id'])
data = data.sort_values(by='Total_reorders', ascending=False)[:10]
print(data)


data = orders_products.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'}).reset_index()
data = pd.merge(data, products[['product_id', 'product_name']], how='left', on=['product_id'])
data = data.sort_values(by='Total_reorders', ascending=False)[:10]
data  = data.groupby(['product_name']).sum()['Total_reorders'].sort_values(ascending=False)
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation='vertical')
sns.barplot(data.index, data.values)
plt.ylabel('Number of Reorders', fontsize=12)
plt.xlabel('Most ordered Products', fontsize=12)
plt.title('which products are ordered the most', fontweight = 'bold')
plt.show()

#most important aisles: best selling
data = orders_prior_all['aisle'].value_counts()
plt.figure(figsize=(12,10))
sns.barplot(data.index, data.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.title('Important Aisles', fontweight ='bold')
plt.xticks(rotation='vertical')
plt.show()

#customer's favorite best selling departments (number of orders)
data = orders_prior_all['department'].value_counts()
plt.figure(figsize=(12,10))
sns.barplot(data.index, data.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Departments wise distribution of products", fontweight='bold')
plt.xticks(rotation='vertical')
plt.show()
