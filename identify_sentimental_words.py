# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:52:03 2019

@author: My
"""
from textblob import TextBlob

def identify_sentimental_words(text):
    for rows in text:         
        blob = TextBlob(rows[5])
        identified_words = []
        for words, tag in blob.tags:
            if (tag in ['JJ','JJS','JJR','RBR','RBS']):
                identified_words.append(words)
    return identified_words
                        