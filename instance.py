import pandas as pd
import instance;

# in this class we define an instance : its same as an row (series) pandas
#------------------------------------------------------
class inst:

    # define params.
    #@param dictionary h : contai,ns set of params with their values.
    
    #---------------------------------------------------
    # Initializer / Instance Attributes
    def __init__(self, s):
        self.h = dict();
        if(not s.empty):
            atts = s.keys()
            for att in atts:
                value = s[att];
                self.addPair(att,value);
    
    #----------------------------------------------------            
    # add a pair to the dictionary.
    def addPair(self, key, value):
        self.h[str(key)] = str(value);        
        return;
    #---------------------------------------------------- 
    # To Str : is method , return the instance as a string.
    def toStr(self , typ, sep="\t"):
        s="";
        if(typ == "key"):
            for key in self.h.keys():
                s = s+ str(key)+sep;
        elif(typ == "value"):
            for key in self.h.keys():
                value = self.h[key];
                s = s+ str(value)+sep;
        elif(typ == "both"):
            for key in self.h.keys():
                value = self.h[key];
                s = s+ str(key)+" = "+str(value)+sep;
            
        return s;
#------------------------------------------------------------

##s = pd.Series([]);
##s['a'] = 'aa';
##s['b'] = 'bb';
##
##it = inst(s);
##print(it.toStr("value"));
