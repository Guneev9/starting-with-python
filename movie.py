#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:30:33 2017

@author: guneevkaur
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetching data and formatting  it
data = fetch_movielens(min_rating =4.0)

print type(data)
#printing training and test data

train = repr(data['train'])
print(train)

test = repr(data['test'])
print(test)

#creating model
model = LightFM (loss='warp')

#training model
model.fit(data['train'], epochs=30,num_threads=2)

def recommendation(model,data,user_ids):
# getting number of users and data items using shape attribute of data dictionary
    n_users,n_items = data['train'].shape
                       
#generating recommendation for each user 
    for user_id in user_ids:
#movies they already like
       known_positives =   data['item_labels'][data['train'].tocsr()[user_id].indices]
#movie our model predict that they will like
       scores = model.predict(user_id, np.arange(n_items))
#ranking in order of most liked to least
       top_items = data['item_labels'][np.argsort(-scores)]



#printing results
       print("user %s" % user_id)
       print("   known positives:")

#top 3 known positives
       for x in known_positives[:3]:
         print("   %s" % x)

       print("    Recommended:")

#top 3 recommended movies
       for x in top_items[:3]:
        print("       %s"   % x)
     
    
# 3 random userids
recommendation(model, data, [3,25,450])









                 
