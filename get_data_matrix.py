# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:36:47 2019

@author: Joan

Decoding json file

* Add number of words in each review!

"""
import time
import json
import numpy as np
import csv
input_bad = open('echobad1.json', 'r')
input_good = open('echogood1.json', 'r')
bad_dict = json.load(input_bad)
good_dict = json.load(input_good)

bad_count = 0
good_count = 0
review_list = []
bad_list = []
good_list = []
n = 409
i = 0

with open('amazon_sitemap_tut.csv', mode = 'r', encoding = 'utf8') as infile:
    
    reader = csv.reader(infile)
    reader.__next__()
    
    # Fecking code block is the problem
#    n = sum(1 for row in reader)
#    print(n)
    X = np.ones((n-1, 4))
    
    time_start = time.time()
    
    for rows in reader: # For each review
        i += 1
        
        bad_count = 0
        good_count = 0
        review_list = rows[5].split()
        #print(review_list)
        for bad_word in bad_dict.keys():
            if bad_word in review_list:
                bad_count += 1
                X[i-1, 2] = rows[6][0]
                X[i-1, 3] = len(review_list)
                #print(bad_word)
        for good_word in good_dict.keys():
            if good_word in review_list:
                good_count += 1
        X[i-1, 0] = bad_count
        X[i-1, 1] = good_count
        
                
    time_tot = time.time() - time_start
    print(bad_count) #123550 
    print(good_count) #128961
    print(time_tot) #77 seconds
    print(bad_list)
    print(good_list)
    print(n)
    print(X)
    
"""
    Results:
    - 5002 reviews
    - 123550 bad words in original reviews
    - 128961 good words in orignial reviews
    - 89 seconds
    
    When bad and good dicts used on another set of reviews (amazon_sitemap_tut.csv),
    - 409 reviews
    - 5161 bad words
    - 5359 good words
    - 4.22 seconds
    
"""








