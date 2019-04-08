# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:05:53 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:25:50 2019

@author: Joan

This program adds all the words in an Amazon review to 2 dictionaries (Good and Bad).

"""
import time
import csv
import json
import re
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
import all_clean_funcs
import operator
from textblob import TextBlob
import emoji

name_of_bad = str(input('Enter file name for bad dictionary:'))
name_of_good = str(input('Enter file name for good dictionary:'))

time_start = time.time()

good = []
bad = []
c=0

with open('all_out.csv', mode = 'r', encoding = 'utf8') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        print(c)        
        # --- STRING --- #
        # Remove all punctuation and symbols
        no_punct = remove_symbols(rows[5])
        
        # Remove emojis
        emojidict = {}
        no_punct_emoji = extractemoji(no_punct, emojidict)
                
        # --- LIST OF WORDS ---#
        # Split string of review into list of words as elements
        #no_punct_split = no_punct.split()
        
        # Remove prepositions, coordinating conjunctions,
        # cardinal number, determiner, to
        grammar_cleaned, grammar_removed = remove_grammar_words(no_punct_emoji)
        
        # Remove stopwords
        filtered_sentence = remove_stop_words(grammar_cleaned)
        
        # Put words in either good or bad dicts
        if rows[6] == '1.0 out of 5 stars' or rows[6] == '2.0 out of 5 stars' or rows[6] == '3.0 out of 5 stars':
            bad.extend(filtered_sentence)
        else:
            good.extend(filtered_sentence)
        c+=1
    
    # Count frequencies of words in dicts
    bad_dict = dict(Counter(bad))
    good_dict = dict(Counter(good))
            
    # Remove words that appear in both dicts
    cle_bad_dict, cle_good_dict = clean(bad_dict, good_dict, 0.3)
    
    # Sort dicts by descending value
    sort_bad_dict, sort_good_dict = sort_dict_desc(cle_bad_dict, cle_good_dict)
    
    print('Number of pairs in bad dict:', len(sort_bad_dict))
    print('\nNumber of pairs in good dict:', len(sort_good_dict))

    print(sort_bad_dict)

save_dicts(name_of_bad, name_of_good, sort_bad_dict, sort_good_dict)

time_elapsed = time.time() - time_start

print('Time Elapsed:', time_elapsed)
