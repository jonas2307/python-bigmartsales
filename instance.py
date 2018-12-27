import pandas as pd

# in this class we define an instance : its same as an row (series) pandas
#------------------------------------------------------
class inst:

    # define params.
    #@param dictionary h : contains set of params with their values.
    
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
    #---------------------------------------------------    
    def isKey(self,key):
        return (key in self.h.keys());        
    #---------------------------------------------------
    def getValue(self, key):
        if(self.isKey(key)):
           return self.h[key];
        return None;
    #----------------------------------------------------
    def getAttributes(self):
        return self.h.keys();
    #----------------------------------------------------
    def items(self,separated=False):
        if(separated):
            return (self.h.keys(), self.h.values());
        return self.h.items();
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

# in this class we define a cluster : 
#------------------------------------------------------
class cluster:

    # define params.
    #@param dictionary G : contains set instances
    #@param center c : represent the cluster center.
    
    #---------------------------------------------------
    # Initializer / Instance Attributes
    def __init__(self):
        self.G = list([]);
        self.c = None;
    #----------------------------------------------------            
    # add an instance to the cluster.
    def addInstance(self, it):
        self.G.append(it);                    
        return;
    #----------------------------------------------------
    def setCenter(self, c):
        self.c = c;
        return;
    #---------------------------------------------------- 
    def size(self):
        return len(self.G);
    #----------------------------------------------------
    def get(self,i):
        if ( i <0 or i> self.size()):
            return None;
        return self.G[i];
    #----------------------------------------------------
    def clear(self):
        self.G=list([]);
        return;
#------------------------------------------------------------
