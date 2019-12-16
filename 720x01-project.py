#!/usr/bin/env python
# coding: utf-8

# In[966]:


import requests
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
from random import random
import math


# ## Subfunctions

# In[933]:


def repos_name(user_data_repos):
    repos = []
    description=[]
    for i in range(len(user_data_repos)):
        
        if user_data_repos[i]['name'] not in repos_test:
            repos.append(user_data_repos[i]['name'])
            
            if user_data_repos[i]['description'] is not None:
                description.append(user_data_repos[i]['description'])
            else:
                description.append(user_data_repos[i]['name'])
        
        else:
            print(user_data_repos[i]['name'])
    
    return repos,description

def fetch_contributor(name, repo):
    name='/'+name
    repo='/'+repo
    url_contributor = 'https://api.github.com/repos'+name+repo+'/contributors'
    requestObj_con = requests.get(url = url_contributor, headers=headers)
    user_data_con = requestObj_con.json()

    if type(user_data_con) == list:
        user_data = user_data_con
        ids = []
        num_cons = []
        names = []
        for i in np.arange(0,len(user_data)):
            ids.append(user_data[i]['id'])
            num_cons.append(user_data[i]['contributions'])
            names.append(user_data[i]['login'])
    else:
        ids=None
        num_cons=None
        names=None
    return ids, num_cons, names

def fetch_committer(user_data):
    list_committer = []
    for i in np.arange(0,len(user_data)):
        if user_data[i]['committer'] is not None:
            list_committer.append(user_data[i]['committer']['id'])
        else:
            list_committer.append(0)
    return list_committer


# In[879]:


#input a person' name, output this person' all repos
def all_repos(name):
    name='/'+name
    url_owner ='https://api.github.com/users'+name
    headers = {'Authorization':'token e392138bc68d0d084dffb5da033d07168b7eafeb'}
    requestObj_owner = requests.get(url = url_owner, headers=headers)
    user_data_owner = requestObj_owner.json()
    url_repos = user_data_owner['repos_url']
    owner_id = user_data_owner['id']
    owner_name = user_data_owner['login']
    requestObj_repos = requests.get(url = url_repos, headers=headers)
    user_data_repos = requestObj_repos.json()
    repos,descriptions = repos_name(user_data_repos)
    return repos,descriptions

# combine repos and their descriptions
def repos_desc(repos, descriptions):
    repos_string=[]
    for i in range(len(repos)):
        if descriptions[i] is not None:
            string_temp = repos[i]+' '+descriptions[i]
        else:
            string_temp = repos[i]
        repos_string.append(string_temp)
    return repos_string

def network(name):
    repos, descriptions = all_repos(name)
    num = len(repos)
    # print(repos)
    edge = {'source':[],'target':[],'weight':[]}
    node = {'id':[],'label':[]}
    url_owner ='https://api.github.com/users'+'/'+name
    headers = {'Authorization':'token e392138bc68d0d084dffb5da033d07168b7eafeb'}
    requestObj_owner = requests.get(url = url_owner, headers=headers)
    user_data_owner = requestObj_owner.json()
    owner_id = user_data_owner['id']
    owner_name = user_data_owner['login']
    node['id'].append(owner_id)
    node['label'].append(owner_name)
    
    for repo in repos:
        #repo = '/'+repo   
        url_commit = 'https://api.github.com/repos'+'/'+name+'/'+repo+'/commits'
        requestObj_com = requests.get(url = url_commit, headers=headers)
        user_data_com = requestObj_com.json()
        if type(user_data_com) != dict:
            ids, num_cons, names = fetch_contributor(name, repo)
            weight_cons = np.array(num_cons)    
            sum_weight_cons = sum(weight_cons)
            weight_cons = weight_cons/sum_weight_cons
            weight_cons = np.round(weight_cons,3)
            for i in range(len(ids)):
                edge['source'].append(owner_id)
                edge['target'].append(ids[i])
                edge['weight'].append(weight_cons[i])
                if ids[i] not in node['id']:
                    node['id'].append(ids[i])
                    node['label'].append(names[i])
    name_list_string = [name]*num
    repos_string={}
    repos_string['repos']=repos
    repos_string['descriptions']=descriptions
    repos_string['name_list']=[name]*num # who fork this repo
    return num, repos, repos_string, edge, node


# In[664]:


#merge 2 dicts with the same keys
def merge_dict(df1,df2):
    df={}
    for key in df1.keys():
        df[key] = df1[key]+df2[key]
    return df


# In[661]:


# save data
'''
network = pd.DataFrame(edge_total)
network.to_csv('net_han.csv',index=False)

network_node=pd.DataFrame(node_total)
network_node.drop_duplicates(['id','label'],keep='first',inplace=True)
network_node.to_csv('net_node_han.csv',index=False)
'''


# ### similarity

# #### SequenceMatcher:
# The idea is to find the longest contiguous matching subsequence that contains no “junk” elements; these “junk” elements are ones that are uninteresting in some sense, such as blank lines or whitespace.
# #### The Levenshtein distance:
# The Levenshtein distance is a string metric for measuring the difference between two sequences. Informally, the Levenshtein distance between two words is the minimum number of single-character edits (i.e. insertions, deletions, or substitutions) required to change one word into the other.

# In[513]:


import re, math
from collections import Counter
from fuzzywuzzy import fuzz
import Levenshtein as lev
import difflib
import statistics

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def cos_sim(s1, s2):
    s1=s1.lower()
    s2=s2.lower()
    vector1 = text_to_vector(s1)
    vector2 = text_to_vector(s2)
    cosine = get_cosine(vector1, vector2)
    return cosine

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def similarity(s1,s2):
    s1=s1.lower()
    s2=s2.lower()
    cosine = cos_sim(s1,s2)
    Ratio = fuzz.ratio(s1.lower(),s2.lower())
    ratio = np.round(Ratio/100,2)
    Distance = lev.distance(s1.lower(),s2.lower())
    dis = np.round(1-Distance/max(len(s1),len(s2)),2)
    SeqMatch = string_similar(s1,s2)
    sim = [cosine, SeqMatch, ratio, dis]
    mean_sim = statistics.mean(sim)
    return mean_sim


# ### similarity test for peport

# In[1024]:


t1='GitHub API Python'
t2='GitHub_API_Python'
t3='GitHub-API-Python'
t4='GITHUB-API-PYTHON'
t5='Python'


# In[1025]:


s1 = lev.distance(t1,t2)
s2 = lev.distance(t1.lower(),t2.lower())
s3 = lev.distance(t1.lower(),t3.lower())
s4 = lev.distance(t1,t4)
s5 = lev.distance(t1.lower(),t5.lower())
s1 = np.round(1-s1/max(len(t1),len(t2)),2)
s2 = np.round(1-s2/max(len(t1),len(t2)),2)
s3 = np.round(1-s3/max(len(t1),len(t3)),2)
s4 = np.round(1-s4/max(len(t1),len(t4)),2)
s5 = np.round(1-s5/max(len(t1),len(t5)),2)
print(s1,s2,s3,s4,s5)


# In[1013]:


s1=string_similar(t1, t2)
s2=string_similar(t1, t3)
s3=string_similar(t1, t4)
s4=string_similar(t1, t5)
print(s1,s2,s3,s4)


# In[993]:


s1=cos_sim(t1, t2)
s2=cos_sim(t1, t3)
s3=cos_sim(t1, t5)
print(s1,s2,s3)


# In[797]:


def weight_cal(num_con):   
    weight_con = np.array(num_con)    
    sum_weight_con = sum(weight_con)
    weight_con = weight_con/sum_weight_con
    weight_con = np.round(weight_con,3)
    return weight_con.tolist()

def list_multiply(List1,List2):
    List = np.multiply(np.array(List1),np.array(List2))
    return List.tolist()


# In[695]:


def similarit_list(repo_new, developer_name):
    num, repos, repos_string, edge, node=network(developer_name) 

    sim = []
    s1 = repo_new
    for j in range(num):
        s2 = repos[j]
        sim.append(similarity(s1,s2))
    return sim, repos


# ## Main function

# In[882]:


# repos and descriptions used to evaluate the model

a='ai_law' 
a_d='all kinds of baseline models for long text classificaiton( text categorization)'

b='fashion-recommendation' 
b_d='A clothing retrieval and visual recommendation model for fashion images'

c='aspnet-core-social-network'
c_d='aspnet-core-social-network'

d='Bitbucket-Api'
d_d='Bitbucket-Api'

e='FDTKit'
e_d='FDTKit is a fuzzy decision tree toolkit includes the state of art fuzzy decision tree algorithms'

repos_test = [a,b,c,d,e]
desc_test =[a_d, b_d, c_d, d_d, e_d]


# ###  1-layer social network - hanwang

# In[880]:


num0, repos0, repos_string0, edge0, node0=network('hanwang921017')
# save data
network0 = pd.DataFrame(edge0)
network0.to_csv('net_han0_e.csv',index=False)

network_node0=pd.DataFrame(node0)
network_node0.drop_duplicates(['id','label'],keep='first',inplace=True)
network_node0.to_csv('net_node_han0_e.csv',index=False)


# In[1028]:


print(repos0)


# ### 2-layer social network - hanwang

# In[872]:


name_0 = 'hanwang921017'
#repo_new = 'movie recommendation python'

num, repos, repos_string_total,edge_total, node_total=network(name_0)
for repo in repos:
    con_ids, num_cons, con_names = fetch_contributor(name_0,repo)
    for name in con_names:
        num, repos, repos_string, edge, node=network(name) 
        edge_total = merge_dict(edge_total, edge)
        node_total = merge_dict(node_total, node)
        repos_string_total = merge_dict(repos_string_total,repos_string)


# In[1035]:


repos_string_total_new={}
for key in repos_string_total.keys():
    repos_string_total_new[key]=[]

for i in range(len(repos_string_total['repos'])):
    repo_new = repos_string_total['repos'][i]
    if repo_new not in repos_string_total_new['repos']:
        for key in repos_string_total.keys():
            repos_string_total_new[key].append(repos_string_total[key][i])
        
print(len(repos_string_total_new['repos']))


# In[873]:


print(type(node_total))
print(len(node_total['id']))


# In[875]:


print(len(repos_string_total['repos']))
for rp in repos_test:
    print(rp in repos_string_total['repos'])


# In[876]:


# save data
network = pd.DataFrame(edge_total)
network.to_csv('net_han_e.csv',index=False)

network_node=pd.DataFrame(node_total)
network_node.drop_duplicates(['id','label'],keep='first',inplace=True)
network_node.to_csv('net_node_han_e.csv',index=False)


# In[955]:


# repos used to evaluate the model

authors=['brightmart','khanhnamle1994','Qolzam','RookieOne','mhjabreel']

a='ai_law' 
a_d='all kinds of baseline models for long text classificaiton( text categorization)'
a2 = 'Hierarchical-Text-Multi-Label-Classificaiton'
a2_d='About Hierarchical Muti-Label Text Classification based on hybrid method (local & global)'


b='fashion-recommendation' 
b_d='A clothing retrieval and visual recommendation model for fashion images'
b2='movie-recommendation-system'
b2_d='A movie recommendation system given by user data, movie data and social data'


c='aspnet-core-social-network'
c_d='aspnet-core-social-network'
c2='react-social-network'
c2_d='Simple React Social Network'

d='Bitbucket-Api'
d_d='Bitbucket-Api'
d2='Minature-Stackoverflow-APIs'
d2_d='The project is containing all the existing features of Stack Overflow and with some additional features and also give connection to the AWS services'

e='FDTKit'
e_d='FDTKit is a fuzzy decision tree toolkit includes the state of art fuzzy decision tree algorithms'
e2='fuzzyLilly'
e2_d='A wrapper to create fuzzy decision trees for more accurate fuzzy predictions'

repos_test = [a,a2,b,b2,c,c2,d,d2,e,e2]
desc_test =[a_d,a2_d,b_d,b2_d,c_d,c2_d,d_d,d2_d,e_d,e2_d]


# In[945]:


url_contributor = 'https://api.github.com/repos'+'/'+'AbdulAli'+'/'+'AbdulAli'+'/commits'
requestObj_con0 = requests.get(url = url_contributor, headers=headers)
user_data_con = requestObj_con0.json()


# In[1074]:


import heapq
def top_k(num):
    
    #repo_new='StackOverfFlow API with Python'
    winner_list=[]
    for repo_i in range(len(repos_test)):
        [repo_new,repo_desc]=[repos_test[repo_i],desc_test[repo_i]]
        s1 = repo_new
        s1_d = repo_desc
        sim_list=[]
        for s2,s2_d in zip(repos_string_total['repos'],repos_string_total_new['descriptions']):
            a_temp = np.round(similarity(s1,s2),5) # similarity(new title, title)
            a_temp = np.round(similarity(s1_d,s2),5) + a_temp # similarity(new decs, title)
            b_temp = np.round(similarity(s1_d,s2_d),5) # similarity(new desc, desc)
            b_temp = np.round(similarity(s1,s2_d),5) + b_temp # similarity(new title, desc)

            sim_list.append(a_temp+b_temp)# sum similarity between titles and descs
        #print('length of similarity list:',len(sim_list))
        '''
        n=copy.copy(num)
        m=copy.copy(num) # number of condidate developers recommended
        while True:    
            index_n = heapq.nlargest(n, sim_list)
            index_n =- np.sort(-np.unique(index_n))
            if len(index_n)<m:
                n=n+1
            else:
                break
        '''
        m=20
        index_n = heapq.nlargest(m, sim_list)
        condidate={'name_fork':[],'repo':[],'description':[],'similarity':[],
                   'id_contributor':[],'name_contributor':[],'weight':[],'score':[]}

        for i in range(m):
            max_i = sim_list.index(index_n[i])

            name = repos_string_total_new['name_list'][max_i]

            repo = repos_string_total_new['repos'][max_i]

            desc = repos_string_total_new['descriptions'][max_i]

            url_contributor = 'https://api.github.com/repos'+'/'+name+'/'+repo+'/commits'
            requestObj_con0 = requests.get(url = url_contributor, headers=headers)
            user_data_con = requestObj_con0.json()

            if type(user_data_con) == list:

                ids_con, num_con, name_con = fetch_contributor(name, repo)

                weight_con = weight_cal(num_con)

                condidate['id_contributor']=condidate['id_contributor']+ids_con
                condidate['name_contributor']=condidate['name_contributor']+name_con

                condidate['weight']=condidate['weight']+weight_con

                num_contributor = len(name_con)

                condidate['name_fork'] = condidate['name_fork']+[name]*num_contributor
                condidate['repo']=condidate['repo']+[repo]*num_contributor
                condidate['description']=condidate['description']+[desc]*num_contributor
                condidate['similarity']=condidate['similarity']+[index_n[i]]*num_contributor

        condidate['score'] = list_multiply(condidate['similarity'],condidate['weight'])    
        #print(condidate)
        
        # merge same contributors
        condidate_new={'name_fork':[],'repo':[],'description':[],'similarity':[],
           'id_contributor':[],'name_contributor':[],'weight':[],'score':[]}

        for i in range(len(condidate['score'])):
            if condidate['id_contributor'][i] not in condidate_new['id_contributor']:
                for key in condidate.keys():
                    condidate_new[key].append(condidate[key][i])
            else:
                rep_ind=condidate_new['id_contributor'].index(condidate['id_contributor'][i])
                condidate_new['score'][rep_ind]=condidate_new['score'][rep_ind]+condidate['score'][i]
            

        #print(condidate['score'])
        score = condidate_new['score']
        '''
        n=copy.copy(num)
        m=copy.copy(num) # number of final developers recommended
        while True:    
            index_n = heapq.nlargest(n, score)
            index_n =- np.sort(-np.unique(index_n))
            if len(index_n)<m:
                n=n+1
            else:
                break
        '''
        
        index_n = heapq.nlargest(num, score)
        #print(index_n)
        max_list=[] # positions of maximum
        for i in range(num):
            max_i = score.index(index_n[i])
            max_list.append(max_i)
        #print(max_list)

        winner ={'name_fork':[],'repo':[],'description':[],'similarity':[],
                   'id_contributor':[],'name_contributor':[],'weight':[],'score':[]}
        for i in max_list:
            for key in winner.keys():
                winner[key].extend([condidate_new[key][i]]) 
        winner_list.append(winner)
        #print('\nrepo_i:',repo_i)
        #print('\nwinner:\n', winner)
    #print(len(winner_list))

    #accuracy
    precision=0
    for i in range(5):
        author=authors[i]
        for j in range(2):
            if author in winner_list[2*i+j]['name_contributor']:
                precision+=10
            #else:print(2*i+j)
    return precision/100,winner_list


# In[1079]:


# save data
win = pd.DataFrame(a[1][0])
win.to_csv('winner_TC_TOP5_1.csv')
win = pd.DataFrame(a[1][1])
win.to_csv('winner_TC_TOP5_2.csv')


# In[1080]:


x = np.arange(1,11)
y=[]
winner_top_k=[]
for k in x:
    p_temp, winner_list_temp=top_k(k)
    y.append(p_temp)
    winner_top_k.append(winner_list_temp)


# In[1081]:


#plot the graph after get the top k precisions
plt.plot(x,y,marker = '*',markersize='10')
plt.xlabel('Top-k')
plt.ylabel('Precision')
plt.xticks(x, x)
plt.show()


# In[1086]:


for i in range(2):
    for j in range(10):
        winner_top_k[i][j]['name_contributor']
print(winner_top_k[0][2])


# In[1083]:


win = pd.DataFrame(winner_top_k[9][0])
win.to_csv('winner_TC_TOP10_1.csv')


# authors=['brightmart','khanhnamle1994','Qolzam','RookieOne','mhjabreel']

# ### test users:
# RookieOne: Stackoverflow API 
# khanhnamle1994: movie recommendation
# Qolzam: social network
# mhjabreel: fuzzy decision tree
# brightmart: text classification

# In[257]:


url_limit = 'https://api.github.com/rate_limit'
headers = {'Authorization':  'token e392138bc68d0d084dffb5da033d07168b7eafeb'}

requestObj_limit = requests.get(url = url_limit, headers=headers)
user_data_limit = requestObj_limit.json()
print(user_data_limit)

