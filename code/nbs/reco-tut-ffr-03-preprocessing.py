#!/usr/bin/env python
# coding: utf-8

# In[ ]:


project_name = "reco-tut-ffr"; branch = "main"; account = "sparsh-ai"


# In[ ]:


get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
import mykeys
get_ipython().system(u'rm /content/mykeys.py')
path = "/content/" + project_name; 
get_ipython().system(u'mkdir "{path}"')
get_ipython().magic(u'cd "{path}"')
import sys; sys.path.append(path)
get_ipython().system(u'git config --global user.email "nb@recohut.com"')
get_ipython().system(u'git config --global user.name  "colab-sparsh"')
get_ipython().system(u'git init')
get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
get_ipython().system(u'git pull origin "{branch}"')
get_ipython().system(u'git checkout main')


# In[ ]:


get_ipython().magic(u'cd "/content/reco-tut-ffr"')


# In[ ]:


import os
import csv
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import networkx as nx
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


# In[ ]:


traindf = pd.read_parquet('./data/bronze/train.parquet.gzip')
traindf.head()


# In[ ]:


# g = nx.from_pandas_edgelist(traindf,
#                             source='source_node',
#                             target='destination_node',
#                             create_using=nx.DiGraph())


# ## Negative sampling

# Generating some edges which are not present in graph for supervised learning. In other words, we are generating bad links from graph which are not in graph and whose shortest path is greater than 2.

# In[ ]:


## This pandas method is super slow, not sure why, compare to csv reader method
# traindf['weight'] = 1
# edges = traindf.set_index(['source_node','destination_node']).T.to_dict('records')[0]

traindf.to_csv('/content/train_woheader.csv', header=False, index=False)
r = csv.reader(open('/content/train_woheader.csv','r'))
edges = dict()
for edge in r:
    edges[(edge[0], edge[1])] = 1


# In[ ]:


missing_edges = set([])
with tqdm(total=9437519) as pbar:
    while (len(missing_edges)<9437519):
        a=random.randint(1, 1862220)
        b=random.randint(1, 1862220)
        tmp = edges.get((a,b),-1)
        if tmp == -1 and a!=b:
            try:
                if nx.shortest_path_length(g,source=a,target=b) > 2: 
                    missing_edges.add((a,b))
                else:
                    continue  
            except:  
                    missing_edges.add((a,b))              
        else:
            continue
        pbar.update(1)


# In[ ]:


list(missing_edges)[:10]


# In[ ]:


# pickle.dump(missing_edges,open('/content/missing_edges_final.p','wb'))


# ## Train/test split

# > Tip: We will split positive links and negative links seperatly because we need only positive training data for creating graph and for feature generation.

# In[ ]:


#reading total data df
df_pos = traindf.copy()
df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])

print("Number of nodes in the graph with edges", df_pos.shape[0])
print("Number of nodes in the graph without edges", df_neg.shape[0])


# In[ ]:


#Train test split 
#Spiltted data into 80-20
X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.2, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.2, random_state=9)

print('='*60)
print("Number of nodes in the train data graph with edges", X_train_pos.shape[0],"=",y_train_pos.shape[0])
print("Number of nodes in the train data graph without edges", X_train_neg.shape[0],"=", y_train_neg.shape[0])
print('='*60)
print("Number of nodes in the test data graph with edges", X_test_pos.shape[0],"=",y_test_pos.shape[0])
print("Number of nodes in the test data graph without edges", X_test_neg.shape[0],"=",y_test_neg.shape[0])


# In[ ]:


# X_train = X_train_pos.append(X_train_neg,ignore_index=True)
# y_train = np.concatenate((y_train_pos,y_train_neg))
# X_test = X_test_pos.append(X_test_neg,ignore_index=True)
# y_test = np.concatenate((y_test_pos,y_test_neg)) 

# X_train.to_csv('train_after_eda.csv',header=False,index=False)
# X_test.to_csv('test_after_eda.csv',header=False,index=False)
# pd.DataFrame(y_train.astype(int)).to_csv('train_y.csv',header=False,index=False)
# pd.DataFrame(y_test.astype(int)).to_csv('test_y.csv',header=False,index=False)


# In[ ]:


# #removing header and saving
# X_train_pos.to_csv('train_pos_after_eda.csv',header=False, index=False)
# X_test_pos.to_csv('test_pos_after_eda.csv',header=False, index=False)
# X_train_neg.to_csv('train_neg_after_eda.csv',header=False, index=False)
# X_test_neg.to_csv('test_neg_after_eda.csv',header=False, index=False)

data_path_silver = './data/silver'
if not os.path.exists(data_path_silver):
    os.makedirs(data_path_silver)

def store_df(df, name):
    df.to_parquet(os.path.join(data_path_silver,name+'.parquet.gzip'), compression='gzip')

# store_df(X_train_pos, 'X_train_pos')
# store_df(X_test_pos, 'X_test_pos')
# store_df(X_train_neg, 'X_train_neg')
# store_df(X_test_neg, 'X_test_neg')
# store_df(X_train, 'X_train')
# store_df(X_test, 'X_test')
store_df(pd.DataFrame(y_train.astype(int), columns=['weight']), 'y_train')
store_df(pd.DataFrame(y_test.astype(int), columns=['weight']), 'y_test')


# In[ ]:


train_graph = nx.from_pandas_edgelist(X_train_pos,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())

test_graph = nx.from_pandas_edgelist(X_test_pos,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())


# In[ ]:


print(nx.info(train_graph))
print(nx.info(test_graph))


# In[ ]:


# finding the unique nodes in both train and test graphs
train_nodes_pos = set(train_graph.nodes())
test_nodes_pos = set(test_graph.nodes())

trY_teY = len(train_nodes_pos.intersection(test_nodes_pos))
trY_teN = len(train_nodes_pos - test_nodes_pos)
teY_trN = len(test_nodes_pos - train_nodes_pos)

print('no of people common in train and test -- ',trY_teY)
print('no of people present in train but not present in test -- ',trY_teN)
print('no of people present in test but not present in train -- ',teY_trN)
print(' % of people not there in Train but exist in Test in total Test data are {} %'.format(teY_trN/len(test_nodes_pos)*100))


# In[ ]:


get_ipython().system(u'git status')


# In[ ]:


get_ipython().system(u'git add .')
get_ipython().system(u"git commit -m 'added silver data layer'")
get_ipython().system(u'git push origin main')

