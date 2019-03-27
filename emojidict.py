import emoji
import re

def extractemoji(string, emojidict):
    noemojistr=''
    for e in string:
        if e in emoji.UNICODE_EMOJI:
            if e in emojidict:
                emojidict[e]=emojidict.get(e)+1
            
            else:
                emojidict.update({e:1})
        else:
            noemojistr+=e
    return noemojistr
            
            
            
string1='this is my solution. This solution removes additional man and woman emoji which cant be renered by python 🤷‍♂ and 🤦‍♀'
print(string1)
dict1={}
string1=extractemoji(string1, dict1)
print(dict1)
print(string1)