# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:36:47 2019

@author: Joan

Decoding json file

* Add number of words in each review!

"""

# No. of reviews in all_out: 23147

import re
import time
import json
import numpy as np
import csv
import pandas as pd
import all_clean_funcs

input_bad = open('allbad.json', 'r', encoding = 'utf-8')
input_good = open('allgood.json', 'r', encoding = 'utf-8')
bad_dict = json.load(input_bad)
good_dict = json.load(input_good)

bad_count = 0
good_count = 0
review_list = []
bad_list = []
good_list = []
exceptions = []
n = 23147
i = 0

with open('all_out.csv', mode = 'r', encoding = 'utf8') as infile:
    
    reader = csv.reader(infile)
    reader.__next__()
    
    X = np.ones((n, 5))
    
    time_start = time.time()
    
    for rows in reader: # For each review
        print(i)       
        bad_count = 0
        good_count = 0
        try:
            rating = int(rows[6][0])
            if rating == 1 or rating == 2 or rating == 3:
                X[i-1, 3] = 0 # Class 0 - bad
                print('it is ', rating)
            elif rating == 4 or rating == 5:
                X[i-1, 3] = 1 # Class 1 - good
            X[i-1, 2] = rating # Rating
                
            review_str = remove_symbols(rows[5])
            review_list = review_str.split()
            
            X[i-1, 4] = len(review_list) # No. of words in that review
            
            for bad_word in bad_dict.keys():
                if bad_word in review_list:
                    bad_count += 1
                    print('0')
            for good_word in good_dict.keys():
                if good_word in review_list:
                    good_count += 1
                    print('1')
                    
            X[i-1, 0] = bad_count
            X[i-1, 1] = good_count
        except:
            exceptions.append(rows[0])
            print(rows[0])
        i += 1
                
    time_tot = time.time() - time_start
    
    df = pd.DataFrame(data = X.astype(int))
    df.to_csv('all_out_mat.csv', sep = ',', header = False, index = False)

    print(bad_count)
    print(good_count)
    print(time_tot)
    print(bad_list)
    print(good_list)
    print(n)
    print(X)
    print('exceptions:', exceptions)
    








