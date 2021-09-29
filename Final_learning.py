#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

raw_dataset = fetch_20newsgroups(subset='all',shuffle=False)
print(raw_dataset.data[0])


# In[2]:


dataset = fetch_20newsgroups(subset='all',shuffle=False,remove=('headers','footers','quotes'))
#print(dataset.target)


# In[3]:


corpus = dataset.data     #total 18846 entries
labels = dataset.target  #total 18846 entries
print(labels)
print(dataset.target_names)


# In[4]:


for i in range(5):
    doc = corpus[i]
    category = dataset.target_names[labels[i]]
    print("The {}-th sent of{}:{}".format(i+1, category, doc))
    print("==============================Here is the doc====================================\n")


# In[5]:


from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from pprint import pprint

def pre_processing(docs):
    tokenizer = RegexpTokenizer(r"\w+(?:[-'+]\w+)*|\w+")
    en_stop = get_stop_words('en')
    for doc in docs:
        raw_text = doc.lower()
        tokens_text = tokenizer.tokenize(raw_text)
        stopped_tokens_text = [i for i in tokens_text if not i in en_stop]
        doc = [token for token in stopped_tokens_text if not token.isnumeric()]
        doc = [token for token in stopped_tokens_text if len(token) > 1]
        # you could always add some new preprocessing here
        yield doc
      
doc1 = remove_stopwords(corpus[0])
doc2 = preprocess_string(corpus[0])
doc3 = next(pre_processing([corpus[0]]))
print(doc2)
print("=========================")
print(doc3)


# In[14]:


docs = list(pre_processing(corpus[:]))
freqs  = defaultdict(int)
for doc in docs:
    for word in doc:
        freqs[word] +=1
print(len(freqs))
vocab = [w for w in freqs if freqs[w]>1]
print(len(vocab))


# In[15]:


from copy import deepcopy

Dict = corpora.Dictionary(docs)

dict1 = deepcopy(Dict)
dict1.filter_extremes(no_below=5, no_above=0.5) #keep_n

low_tf_token = [w for w in freqs if freqs[w]<=3]
remove_ids = [Dict.token2id[w] for w in low_tf_token]
print(len(remove_ids))
Dict.filter_tokens(remove_ids)
Dict.compactify()
print(dict1)


# In[16]:


pprint(Dict.token2id)


# In[17]:


import gensim
corpus1_bows = [Dict.doc2bow(doc) for doc in docs]
corpus2_bows = [Dict.doc2bow(doc) for doc in docs]
news_corpus = [[w for w in doc if w in Dict.token2id] for doc in docs]   #fliter out
print(corpus1_bows[0])
print(news_corpus[0])
lda = LdaModel(corpus1_bows, num_topics=20)


# In[ ]:




