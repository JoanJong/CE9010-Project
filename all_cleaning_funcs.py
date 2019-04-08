# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:07:18 2019

@author: Joan

This code contains all functions for cleaning of dictionary.

"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:25:50 2019

@author: Joan

This program contains all functions for cleaning diciontary and saving to json file.

"""
def remove_symbols(review_list):
    no_punct = re.sub(r'[.!,\[\]\$()\?\*\/\']', '', rows[5]).lower()
    return no_punct

def remove_stop_words(review_list):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in review_list if not w in stop_words]
    return filtered_sentence
    
def clean(bad_dict, good_dict, thresh):
    thresh = thresh
    for bad_key, bad_val in list(bad_dict.items()): # Need to change to list otherwise gives 'dictionary changed size during iteration' error
        for good_key, good_val in list(good_dict.items()):
            if bad_key == good_key:
                if bad_val > good_val:
                    # Want to make sure the difference is big enough to keep bad_key
                    # If difference not bbig enough, treat them as same freq and delete both
                    if thresh * bad_val < good_val:
                        del bad_dict[bad_key]
                        del good_dict[good_key]
                    elif thresh * bad_val > good_val:
                        del good_dict[good_key]
                elif good_val > bad_val:
                    # Want to make sure the difference is big enough to keep good_key
                    # If difference not big enough, treat them as same freq and delete both
                    if thresh * good_val < bad_val:
                        del good_dict[good_key]
                        del bad_dict[bad_key]
                    elif thresh * good_val > bad_val:
                        del bad_dict[bad_key]
                else:
                    # Common word between the 2 dictionaries
                    del bad_dict[bad_key]
                    del good_dict[good_key]
                    
                    continue
            else:
                continue
    return bad_dict, good_dict

# Remove prepositions, coordinating conjunctions,
# cardinal number, determiner, to
# Needs to take in string! Returns list
def remove_grammar_words(review_str):  
    # Create textblob object
    blob = TextBlob(review_str)
    blob_list = []
    blob_list_removed = []
    for words, tag in blob.tags:
        if (tag not in ['IN','CC','CD','DT','TO']):
            blob_list.append(words)   
        else:
            blob_list_removed.append(words)
            
    return blob_list, blob_list_removed

# Remove emojis from string & puts all emojis
# that appear in a new dict
def extractemoji(string, emojidict):
    noemojistr=''
    for e in string:
        if e in emoji.UNICODE_EMOJI:
            if e in emojidict:
                emojidict[e]=emojidict.get(e)+1
            
            else:
                if e == "♂" or e== "♀" :
                    continue
                else:
                    emojidict.update({e:1})
        else:
            noemojistr+=e
    return noemojistr

def sort_dict_desc(bad_dict, good_dict):
    sort_bad_dict = dict(sorted(bad_dict.items(), key = operator.itemgetter(1), reverse = True))
    sort_good_dict = dict(sorted(good_dict.items(), key = operator.itemgetter(1), reverse = True))
    return sort_bad_dict, sort_good_dict

def save_dicts(name_of_bad, name_of_good, bad_dict, good_dict):
    bad_dict_json = name_of_bad + '.json'
    good_dict_json = name_of_good + '.json'
    with open(bad_dict_json, 'w') as fp:
        json.dump(bad_dict, fp)
    with open(good_dict_json, 'w') as fp:
        json.dump(good_dict, fp)
