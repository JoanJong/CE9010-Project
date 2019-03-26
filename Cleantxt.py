def clean(dict1, dict2, thresh=0.25):
    for key1,val1 in dict1:
        for key2, val2 in dict2:
            if key1==key2:
                if val1<val2:
                    a=val1
                    b=val2
                    c=key1
                    d=key2
                else:
                    a=val2
                    b=val1
                    c=key2
                    d=key1
                if a<thresh*b:
                    del clean[c]
                elif a>(1-thresh)*b:
                    del clean[d]
                else:
                    # Common word between the 2 dictionaries
                    del clean[c]
                    del clean[d]
                continue
            else:
                continue
            
            
            
    
    
    
                
