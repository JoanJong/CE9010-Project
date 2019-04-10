# CE9010-Project
Codes:
1) add_to_dicts.py adds words in scraped reviews to 2 dictionaries, good and bad.
2) echobad1.json and echogood1.json are the bad and good dictionaries respectively for a single product scraped.
     - Number of pairs in bad dict: 6474
     - Number of pairs in good dict: 10957
3) emojidict.py cleans string of emoji && updates emoji dictionary with frequency.
4) clean.py cleans good and bad dictionary of redundant words and accidental words.
5) ~~get_data_matrix.py gives data matrix from reviews.~~
   * ~~Column 1 (no. of bad words), Column 2 (no. of good words), Column 3 (no. of rating stars), Column 4 (no. of words in each review)~~
6) identify_sentimental_words.py takes out adjectives and other sentiment-carrying words
7) get_data_matrix.py (modified) gives data matrix from reviews.
   * Header: bad wordcount [0], good wordcount [1], rating [2], class (0 for bad or 1 for good) [3], review wordcount [4]
8) create_and_clean_dicts.py integrates all cleaning functions to create and clean dictionaries.
9) all_clean_funcs.py is a collection of all cleaning functions.
10) allbad.json and allgood.json are dictionaries created from 23147 reviews.
11) all_out_mat.csv is the csv file of the data matrix created from get_data_matrix.py.
12) logreg_supervise_classification is the logistic regression classification
