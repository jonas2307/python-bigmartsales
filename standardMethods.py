import re
import random
import numpy as np


#----------------------------------------------------
# this method check if an string value is a float or not
# value : the element in question.
# Return : True if its float, False otherwise.
#-----------------------------------------------------
def IsFloat(value):
    try:
        float(value); return True;
    except:
        return False
#--------------------------------------------------------

# this method print a list of elements,
# L : the list
# typ : the elements list type's
# sep : used as separator between elements list.
#-------------------------------------------------------
def Print(L,typ, sep="\n"):
    # if sep is null , we consider it as "\n"
    if(sep == None):
        sep = "\n";

    # init the text.
    text = "";
    #--------------------------------------------------
    # if the type is simple list
    if(typ.lower() == 'simple'):
        for i in range(0,len(L)):
            element = L[i];
            text = text + str(element) +sep;
            if(i % 10 == 9):
                print(text); text="";
        print(text); text="";
    #--------------------------------------------------
    # if the type is items list
    elif(typ.lower() == 'items'):
        for i in range(0,len(L)):
            element = L[i]
            (key,value) = element;
            text = text + "["+str(key)+" , "+str(value)+"]"+sep;
            if(i % 10 == 9):
                print(text); text="";
        print(text); text="";
    #-------------------------------------------------
    # if the type is instance list
    elif(typ.lower() == 'instance'):
        for i in range(0,len(L)):
            it = L[i]
            text = text + it.toStr("value")+ sep;
            if(i % 10 == 9):
                print(text); text="";
        print(text); text="";
    #-------------------------------------------------
        
    return;
#-----------------------------------------------------------------------

# get the row as an string
# t : this represent our Series (pandas) that we want to parse it string.
# h : represent the header elements.
# sep : the separator between will be between t elemnts 
#--------------------------------------------------------
def toStr(t, h, sep="\t"):
    s="";
    #print(h," len : "+str(len(h)))
    for i in range(0,len(h)):
        key = h[i];
        key=re.sub('\t','',key)
        if(len(key) == 0):
            continue;
        value = t[key];
        s = s + str(value)+sep;        
    return s;
#--------------------------------------------------------


# this method print a dictionary and return it as text format (string)
# d : the dictionary that we want to print it
# sep1 : the separator between the key and value of an item.
# sep2 : the separator between two items.
# divide : if ittrue, so we print dictionary by part, otherwise we print whole dict at end.
# Return : the dictionary as string variable.
#-----------------------------------------------
def PrintDic(d, sep1="\t",sep2="\n",divide=False):
    s=""; text="";
    i=0;
    for (key,value) in d.items():
        text = ""+text +str(key)+sep1+str(value)+sep2;
        i = i+1;

        if(divide and i % 5 == 0):
            print(text);
            s = s + text;
            text="";

    print(text);s = s + text ; text="";

    return s;
#-----------------------------------------

# this method allow us to (over)write a text in file
# text : the text that we want to write it.
# path : the file path.
# append : if it is true , so we write in the end of file,
#          if it is false, so we overwrite.
#-------------------------------------------
def WriteOnFile(text,path,append=False):
    f=1;
    if(append):
        f= open(path,'a');
    else:
        f= open(path,'w');
    f.write(text);    
    return;
#-------------------------------------------------


# this method is used to get a beautiful printing.
# text : the text that we want to print it.
# key : is the header element.
# Return set of new text
#-------------------------------------------------
def TextWithHeaderToWrite(text,key):
    s="---------"+str(key)+"------------------\n";
    s= s + text;
    s = s+ "\n----------------------------------\n";
    return s;
#---------------------------------------------------

# this method allow us to remove a set of (index)
# L : the set of data !
# I : the set of index
# R : list contains the remaining values
#--------------------------------
def Remove(L,I):

    # sort the list 'L'
    I.sort(reverse=True);
    # build a numpy arry 
    K = np.array(L);
    for i in range(0,len(I)):
        ind = I[i];
        K = np.delete(K,ind);
        
    return list(K);
#----------------------------------------------------
    
##l=[10,15,12,10,12,11];
##L=[0,4, 2];
##print(Remove(l,L));
