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


import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import networkx as nx


# In[ ]:


traindf = pd.read_parquet('./data/bronze/train.parquet.gzip')
traindf.head()


# In[ ]:


traindf.describe()


# In[ ]:


print(traindf[traindf.isna().any(1)])
print(traindf.info())
print("Number of diplicate entries: ",sum(traindf.duplicated()))


# In[ ]:


g = nx.from_pandas_edgelist(traindf,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
print(nx.info(g))


# In[ ]:


subgraph = nx.from_pandas_edgelist(traindf.head(20),
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
# https://stackoverflow.com/questions/9402255/drawing-a-huge-graph-with-networkx-and-matplotlib

pos = nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")
print(nx.info(subgraph))


# In[ ]:


# No of Unique persons 
print("The number of unique persons",len(g.nodes()))


# In[ ]:


# No of followers of each person
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()


# In[ ]:


indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()


# In[ ]:


# No Of people each person is following
outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()


# In[ ]:


print('No of persons who are not following anyone are {} ({:.2%})'.format(sum(np.array(outdegree_dist)==0),
                                                                        sum(np.array(outdegree_dist)==0)/len(outdegree_dist)))


# In[ ]:


print('No of persons having zero followers are {} ({:.2%})'.format(sum(np.array(indegree_dist)==0),
                                                                        sum(np.array(indegree_dist)==0)/len(indegree_dist)))


# In[ ]:


count=0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0 :
        if len(list(g.successors(i)))==0:
            count+=1
print('No of persons those are not following anyone and also not having any followers are',count)

