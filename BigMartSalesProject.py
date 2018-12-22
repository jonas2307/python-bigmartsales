#---------------- Edited by Younes DELHOUM -----------
# Note : this project is in progress ... (not complete yet )
#        Update the paths name depend on your files !
#------------------------------------------------------

# Python Version 3.5.4
# Date : 15/12/2018.
# Main Library Used : Numpy, Pandas, Matplotlib, Sklearn
# KeyWords : Machine Learning, Data Processing.
# Main Method , K-means , Apriori, Decision Trees

# Objective : BigMartSales Project.
# Description :
#   the aim of this project is to use the different pythons libs
# - read dataset (csv file) contains the data (for training, test)
# - do some statical computation (metric values), box plot
# - apply the unsupervised learning algorithm : K-means
# - apply the supervised learning algorithm : Decision trees.
# - get association rules based on Apriori algorithm.


#------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk, re
import pymongo
import urllib
import graphviz
import random

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
    
from scipy import special , optimize
from sqlalchemy import create_engine
from apyori import apriori

from pymongo import MongoClient
from pprint import pprint
from bs4 import BeautifulSoup
from collections import Counter

#--- my scripts :: ----------------------
import standardMethods as sm;
import instance;


#------------------ Project Bigmart-Sales -------------


# this method read an inut file, and return his contents
# path : the input file path.
# typeFile : the file type (csv , json, excel)
# Return : pandas dataframe
#--------------------------------------------------------
def ReadFile(path, typeFile):
    data=None;
    
    if(typeFile.lower() == 'csv'):
        data = pd.read_csv(path);
    elif(typeFile.lower() == 'json'):
        data = pd.read_json(path);
    elif(typeFile.lower() == 'excel'):
        data = pd.read_excel(path);
    
    return data;
#-----------------------------------------------------

# 
# row : set of data
# Return : row as list.
#------------------------------------------------------
def GetHeaderTypes(row):
    header=list(row);
    return header;
#------------------------------------------------------

# this method return a the attributes types as a list.
# h : set of attributes.
# data : a pandas dataframe contains the main data.
# Return : list of string (types : numerical / nominal).
#------------------------------------------------------
def TypeHeader(h,data,toPrint):
    # define an initial dictionary
    d= dict();
    # get first row of data
    row = data.loc[0,h];
    # for each attribut
    for i in range(0,len(h)):
        key=h[i];
        # get the first element
        value=row[key];
        # get the type of the attribut, add it to dictionary
        if(sm.IsFloat(value)):
            d[key]='numerical';
            if (toPrint):
                print(value," is numerical ");
        else:
            if (toPrint):
                print(value," is nominal ");
            d[key]='nominal';
    return d;
#-----------------------------------------------------

# this method return only the attributes which are from type (typ)
# d : contains the type of attributes.
# type : a data type that we want (nominal , numeric)
# Return : the attributes with 'typ' 
#-----------------------------------------------------
def GetSelect(d,typ):
    # init attributes list
    T = list([]);
    # for each element chck if value (t) is equals to typ
    for (key,t) in d.items():
        if(t.lower() == typ.lower()):
            T.append(key);
    return T;
#--------------------------------------------------------
# this method, select only the data (columns) which are from type ('typ')
# data : the set of data which we want to filter based on typ.
# d : contains the type of attributes.
# type : the selected type data (nominal , numeric)
# toPrint : if its true , so we will print some data.
# Return : the selected data.
#-------------------------------------------------------
def SelectData(data,d,typ,toPrint):
    # get only the set of attributes which are from the type (typ)
    T = GetSelect(d,typ);
    if(toPrint):
        print(T);

    # get the set of data satisfied the constraints
    D = data.loc[:,T];
    if(toPrint):
        print(D);    
    return D;
#-------------------------------------------------------


# To Update ...
# this method allow ut to print the data set , row by row
# data : we consider data as pandas dataframe (we extends method later)
# startIndex : references the first (index) element to print.
# endIndex : references the last (index) element to print.
#----------------------------------------------------------------
def PrintByRow(data,startIndex,endIndex, sep="\t"):
    
    # if sep is null , we consider it as "\t"
    if(sep == None):
        sep = "\t";
    # init the text.
    text = "";
    # get the set of attributes (columns) 
    atts = data.columns;
    #print("atts : "+str(len(atts)));
    # update the start, end index.
    if(startIndex == None):
        startIndex = 0;
    if(endIndex == None):
        endIndex = len(data);

    # start by print the header (attributes)
    sm.Print(atts,"simple",sep);
    atts = list(atts);
    
    #for each row
    for i in range(startIndex,endIndex):
        #get the row!
        row = data.loc[i,atts];
        # get the row as an string
        s = sm.toStr(row, atts, sep);
        text = text + str(s) +"\n";
        if(i % 10 == 9):
            print(text); text="";
        print(text); text="";
        
    return;
#-----------------------------------------------------------------------

# this method , generate the dictionary of classes.
# data : represent our data (numeric one)
# nbrClass : the number f class that we want to generate them.
# sep : it is used to create the class format
# Return : dictionary as keys : classes , as values (frequency = 0 because its empty)
#----------------------------------------------------------------------
def InitSet(data, nbrClass, sep='_'):
    L=dict();
    # we add one to cover all elements ! (used more in other methods)
    vMin = min(data); vMax = max(data)+1;
    # if we have only one class
    if(nbrClass == 1):
        #class will be (min_max)
        key=str(vMin)+sep+str(vMax); L[key]=0;
    else:
        #if we have more than one classes ,
        # we compute the step (range length)
        step = (vMax - vMin)/nbrClass;
        # for each classes
        for i in range(0,nbrClass):
            # define the extremities (a and b)
            a = vMin + (i * step); b = a + step;
            # define the class name, and init the frequency value as 0
            key=str(a)+sep+str(b); L[key] = 0;

    # return the dictionary
    return L;
#-----------------------------------------

# this method check if it value is in the range of clas
# clas : the main class.
# it : the element value that we check if he is(not) in this 'clas'.
# sep : the separator used to divide clas in her extrems values (smallest, highest)
# Return : the class if it is defined in L (ranges) , None otherwise
#-----------------------------------------
def IsIn(clas , it, sep):
    # clas : should be like (a_b) where a and b are extrems values of class (range)
    # based on separator divide classname on array
    T = clas.split(sep);
    # if there not 2 elements , print error !
    if(len(T) !=  2):
        print("check your class it should contains 2 elements but it contains"+str(len(T)));
    else:
        #get the extremities values
        a = float(T[0]); b = float(T[1]);
        # if it is in range [a,b[ , return true
        if(it>= a and it < b):
            return True;    
    return False;
#-----------------------------------------

# this method based on the dictionary L , return the class of 'it'.
# it : the element value that we want to get his class.
# sep : the separator used to get the key (class)
# Return : the class if it is defined in L (ranges) , None otherwise
#-----------------------------------------
def GetClass(L, it , sep):
    #for each class of L
    for clas in L.keys():
        # check  if it is in clas, if true , return this class
        if (IsIn(clas , it, sep)):
            return clas;
    #if it is not in any class, return None
    return None;
#-----------------------------------------

# this method compute for each class his elements (size)
# data : the data (column data) , that we want to do some count on them 
# nbrClass : the number of classes
# sep : the separator used to generate the classname.
# Return : the dictionary as string variable.

#---------------------------------------
def DoCount(data , nbrClass, sep):
    #('a_b', f)
    #print(data)

    # get the initi dictionary (with classes as keys)
    L=InitSet(data, nbrClass,sep);

    # for each record
    for i in range(0,len(data)):
        # get the item
        it = data[i];
        #get the class
        c = GetClass(L,it, sep)
        # if the class is existing
        if(c != None):
            # update the frequency and the L dictionary
            freq = L[c]; freq = freq +1; L[c] = freq;
    #return th dictionary
    return L;
#-----------------------------------------

# this method allow us compute and return the metric values.
# (min,max,mean,q1,q3,median) in numerical case, mode in nominal case
# data : the set of records.
# isNm : if true , so data type is numerical
# Return : return the dictionary of values
#-----------------------------------------
def GetMetricValues(data, isNum):   

    d=0;
    
    if(isNum):
        d={ 'mean' : data.mean(),
            'min' : data.min(),
            'max' : data.max(),
            'median' : data.median(),
            'Q1' : data.quantile(0.25),
            'Q3' : data.quantile(0.75)
            };
        
    else:
        d={ 'mode' : data.mode()
            };

    return d;
#-----------------------------------------

# this method allow us to do some statistical operations
# data : the set of records.
# d : list contains the type of attributes (nominal,numeric)
# nbrClass :
# sep : 
# path : the path of destination file. (if its not null)
#------------------------------------------
def DoStats(data , d, nbrClass=2, sep='_', path=None):

    #get the set of attributs
    cols = list(data.columns); print(cols);
    #if path is not null , so we open it
    if(path != None):
        sm.WriteOnFile("",path,False);

    # for each attribute (key)
    for key in cols :
        #if the attribute is a nominal one
        if(key in d.keys() and  d[key].lower() == 'nominal'):
            # get the column data
            dCol=data[key];
            # the counter ! (call the method)
            co = Counter(dCol);
            #get items as a list
            L = list(co.items());
            print("for the attribut : "+key);

            # print the set of items !
            sm.Print(L,"items","\n");

            # get metric values , in case of nominal we get only the mode
            di = GetMetricValues(data[key] , False);
            # get the differents values as a string
            s = sm.PrintDic(di,"\t","\n",True);

            # add the header of the text (s)
            s= sm.TextWithHeaderToWrite(s,key);

            # write it
            if(path != None):
                sm.WriteOnFile(s,path,True);

        # if attribute is a numeric one    
        elif(key in d.keys() and d[key].lower() == "numerical"):

            #check the class count, if its null , put it as 1
            if(nbrClass == None):
                nbrClass = 1;
            print("for the attribut : "+key);
            # get the column data
            dCol=list(data[key]);

            # call the method do Count to get the classes with their frequency
            L = DoCount(dCol, nbrClass, sep);
            #print the dictionary (contains classes , freq)
            sm.PrintDic(L,divide=True);

            # get the metric values of the column data (min, max, mean, median, q1, q3)
            di = GetMetricValues(data[key] , True);
            # dict to string (s)
            s = sm.PrintDic(di,"\t","\n",True);

            # add the header
            s= sm.TextWithHeaderToWrite(s,key);
            if(path != None):
                sm.WriteOnFile(s,path,True);
    
    return;
#-------------------------------------------

# this method allow us to generate box plot from data and save it (using plot of matplotlib)
# data : the set of records that we want to draw their box plot.
# path : the path of destination file.
#-------------------------------------------
def BoxPlot(data, path):
    #generate the boxplot of data
    bx = data.boxplot();
    # if path is null , create a temporary path
    if(path == None):
        path="./tmp_file.png";

    #save the picture
    plt.savefig(path);
    #clear the plot
    plt.clf();
    return;
#------------------------------------------

# this method allow us to generate box plot for each attribute of the data
# data : the set of records that we want to draw their box plots.
# picFolder : the directory where we will store about picture (of boxplot).
#--------------------------------------------
def BoxPlotAllAtts(data, picFolder):
    # get the set of attributes
    cols = data.columns;
    # for each attribute
    for att in cols:
        # data attribute (the column)
        dAtt=pd.DataFrame(data.loc[:,att],columns=[att]);
        # generate the file path (where we will store this boxplot)
        pathFig=picFolder+"/boxplot_"+att+".png";
        #call the customize BoxPlot method, draw figure and save it.
        BoxPlot(dAtt,pathFig);
        
    return;
#------------------------------------------

# test version , not a complete one (15/12/2018 13:40)
# in this method we will use special Kmeans Algorithm (from sklearn library)
# Kmeans algorithm to get the data groupes (clusters).

# data : our data, its a numeric data
# toPrint : if true, we print some data.
# Return : data as set of groups.

#suggeston :
# * change the alogorithm to accept nominal values( specially) for distance computation
# * possible to pass cluster number as param.
#---------------------------------------------------
def KmeansAlgorithm(data, toPrint):

    nbr_clus = 4;
    
    #YOUNES  : what is that ?
    random_state = 170;

    # call Kmeans algorithm of sklearn, user 4 cluster (can change that later).
    # and get the predicted cluster !
    y_pred = KMeans(n_clusters=nbr_clus, random_state=random_state).fit_predict(data)
    if(toPrint):
        print("y : ",y_pred.size);

    # the cluster ! , so now we can print only data are from cluster !
    # add the class to dataframe
    key = 'class';
    data[key] = y_pred;
    #get the list of clusters
    L=list(set(y_pred));

    # divide the data based on key
    G = data.groupby(key);
    
    if(toPrint):
        for e in L:
            print(G.get_group(e));
            
    #return set of group, grouped by class !
    return G;
#-------------------------------------------------
# test version , not a complete one (15/12/2018 13:36)
# in this method we will use special Apriori algorithm to get the association rules.
# data : our data, its a nominal data.
# toPrint : if true, we print some data.
# Return : set of association rules !
#--------------------------------------------------
def AprioriAlgorithm(data, toPrint):
    print(" --- Apriori Algorithm --- ");
    
    #transform data to list of list :
    records=DataFrameToListOfList(data,False);

    if(toPrint):
        print("record len : ",len(records));

    if(toPrint):
        for i in range(0,5):
            print(records[i]);
    #now appply A priori algorithm.
    min_sup=0.15; min_conf=0.5; min_len=3; min_lift=3;

    #configuration set
    config = [min_sup, min_conf, min_len, min_lift];
    if(toPrint):
        print(config);
    
    #call the apriori algorithm (with configuraton settings) and get the association rules.
    AR = apriori(records,
                 min_support=min_sup,
                 min_confidence=min_conf,
                 min_length=min_len,
                 min_lift=min_lift);

    #convert as list
    ARes=list(AR); #print(ARes);

    # print rules
    if(toPrint):
        print(len(ARes));
        for rule in ARes :
            s = RuleToStr(rule);
            print(s);
    # return the list of association rules    
    return ARes;
#--------------------------------------------------------------

# this method return printing format of an association rule.
# rule : the association rule in question,
# Return : the string format of rule (contains , the items, support, confidence and lift)
#---------------------------------------------------------------
def RuleToStr(rule):
    # define the variable
    s="";
    # first index of the inner list
    # Contains base item and add item
    pair = rule[0]; 
    items = [x for x in pair]
    # add the contains to s
    s = s+("Rule: " + str(items[0]) + " -> " + str(items[1]))+"\n";

    #second index of the inner list
    s = s+("Support: " + str(rule[1]))+"\n";

    #third index of the list located at 0th
    #of the third index of the inner list

    s = s+ ("Confidence: " + str(rule[2][0][2]))+"\n";
    s = s+ ("Lift: " + str(rule[2][0][3]))+"\n";
    s = s+ ("=====================================")+"\n";

    return s;
#-------------------------------------------------

# this method convert pandas dataframe to a list to list (it use for Apriori for example)
# data : our original daataframe.
# toPrint : if its true , we print some data
# Return : list of lis
#--------------------------------------------------
def DataFrameToListOfList(data, toPrint):
    # get the set of attributes of data.
    header=GetHeaderTypes(data[0:1]);

    # transform dataframe on numpy matrix
    D = data.as_matrix()
    # init our List of List
    L=list([]);
    # for each record
    for i in range(0,len(data)):
        # init inner list, save record in row variable
        l=[]; row=D[i];
        
        if(i % 100 == 0 and toPrint):
            print(i,row);
        # add element value (value of record , attribut) to inner list
        for j in range(0,len(row)):
            l.append(row[j]);

        #add inner list to Global List
        L.append(l);
        
    # return list of list.    
    return L;
#--------------------------------------------------
# this method reference the class, by new referencing name,
# for example if we want to divide set of numeric data in nbr of classes,
# based in range, we replace first 'range 1' : "0_20" by att_C1 etc.
# att : is the attribut name of data,
# L is list of original classes (in this case , list of ranges)
#--------------------------------------------------
def GetReferencesNames(att, L):
    # initialy we create a dictionary
    h=dict()
    # for each class, create the new class name
    for i in range(0,len(L)):
        classname=att+"_C"+str(i+1);
        key = L[i];
        # reference the classname by the key
        h[key] = classname;
    return h;
#---------------------------------------------------

#--------------------------------------------------
# the aim of this  method is to take  give for each data element,
# with a numeric value , a nominal class,
# data : set of records, (an array)
# att : the data attribut name
# nbrClass : the number of classes that we want to attribute for record,
# sep : the separator is used to define the keys (class_name)
# Return : a nominal data representing the numeric original data.
#----------------------------------------
def NominalData(data,att,nbrClass,sep):
    # define an numpy array to store the classified data
    S = np.array([]);

    # get the set of class depend on data
    L=InitSet(data, nbrClass,sep);

    # get new names :
    h = GetReferencesNames(att , list(L.keys()));

    print(h);
    # get the name depend on the attribt name,
    # we can change it later (if user want a specific name)
    # we can use a ditionary to reference ! or pass it by params
    
    #print("L",L); print("--------------");

    # for each element data
    for i in range(0,len(data)):
        #get the value 
        it = data.iloc[i];
        # get the class
        c = GetClass(L,it, sep);
        # if the class is not null , add the class to the array
        if(c != None):
            #get the name from dict 'h'
            c= h[c];
            # add the new class to array S
            S = np.append(S, [c]);
    
    print(S)
    # rename classes ! by index !
    return S;
#----------------------------------------

# this meth convert the numeric data to a nominal one,
# data : is our numeric data,
# nbrClass : its the number of classes that we want to attribute for new data,
# sep : the separator is used to define the classes
# Return : Nominal Data
#----------------------------------------
def NumericToNominal(data, nbrClass ,sep):
    # create a pandas dataframe (empty)
    D=pd.DataFrame();
    # get the header of data (the attributes
    h= GetHeaderTypes(data[0:1]);
    print(h);

    # for each attribute
    for att in h:
        # get his data
        dCol = data[att];
        # nominal it
        dCol = NominalData(dCol,att, nbrClass,sep);
        #update the dataframe
        D[att] = dCol;
    #return the nominal data
    return D;
#---------------------------------------------------

# this method take two dataframe and return their concatination by columns
# d1 : the first dataframe
# d2 : the second dataframe
# Return : dataframe join both dataframes.
#---------------------------------------------------
def JoinByColumns(d1,d2):
    data = d1;
    for col in d2.columns:
        data[col] = d2[col];
    return data;
#---------------------------------------------------

# this method allow us , to transform data from object (string) to a label one
# this type of data is necessery in case of using decision trees.
# data : our dataframe.
# isSerie : if its true (data is a pandas.Series) , we transform data.
# Return : transformed data.
#--------------------------------------------------
def TransformData(data,isSerie=False):
    print("------------------");
    # call te preprocessing labelencoder (of sklearn)
    le = preprocessing.LabelEncoder();

    # if data is a series        
    if(isSerie):
        if data.dtype == type(object):
            data = le.fit_transform(data);
    else:
        for col in data.columns:
            if data[col].dtype == type(object):
                data[col] = le.fit_transform(data[col])
    return data;
#--------------------------------------------------

# this method, will use the train data to generate the model (classifier),
# use the test data to test the model.
# dTrain : training data.
# dTest : test data.
# clas : the attribute class , which used to create classifier.
# Return : the classifier (predict results (to-update) depend on what user want)
#-------------------------------------------------
def DecisionTree(dTrain, dTest, clas, pathOutGraph):
    #training data
    data = dTrain;
    
    # update step ! replace the numeric values by the mean
    # update step : replace the nominal value by mode.
    dTy=TypeHeader(data.columns,data,False);
    data = ReplaceNaNValues(data,dTy);
        
    # drop nan data (to update this step)
    data = data.dropna();
    print(len(data)); 
    
    # get the target :
    T = data[clas];

    #remove target column
    data = data.drop(columns = [clas]);
    
    # transform data from object (string) to a label ! 
    data = TransformData(data);

    # transform data from object (string) to a label
    T=TransformData(T,isSerie=True);

    # call decision trees method.
    # classifier
    clf = tree.DecisionTreeClassifier();
    #create feed the classifier used data (train) and Targets
    clf = clf.fit(data, T)

    #drop nan data (to update)
    dTest = dTest.dropna();
    
    # transform data, to be able to use it with classifier !
    dTest = TransformData(dTest);
    #dTest = dTest.drop(columns = [clas]);

    # the predict targets !
    Y = clf.predict(dTest);
    #print(Y);

    #print(the test data with class)
    dTest[clas] = Y;

    print(dTest);

    # merge

    # use graphiz to draw tree. # use GVEDit to Visualize the tree.
    # skip it (it doesn't work , (graphviz executables not in path ! )
    #pathOut="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/graphFile.vz";
    dot_data = tree.export_graphviz(clf, out_file=None); 
    graph = graphviz.Source(dot_data); 
    graph.render(pathOutGraph,view=True);

    #print(graph);

    return Y;
#----------------------------------------------------
# ------------ update : 19/12 -----------------------

# this method create an instance based on data (row)
# row : the data.
#--------------------------------------------------
def RowToInstance(row):
    # create  a pandas series object
    s = pd.Series(row);
    # call the inst class and return the instance created.
    it = instance.inst(s);
    return it;
#-------------------------------------------------------------

# this method allow us to convert data on set of instance
# data : we consider data as pandas dataframe (we extends method later)
# startIndex : references the first (index) element to transform on instance.
# endIndex : references the last (index) element to transform on instance.
#----------------------------------------------------------------
def DataToInstances(data,startIndex,endIndex):
    I=list([]);    

    # update the start, end index.
    if(startIndex == None):
        startIndex = 0;
    if(endIndex == None):
        endIndex = len(data);

    # get the set of attributes (columns) 
    atts = list(data.columns);
    print(atts)

    # update the index (to forbidden the key multipte (case of boostrap data)
    index = np.arange(len(data));
    data.index = index;
    
    #for each row
    for i in range(startIndex,endIndex):
        #get the row!
        index = list(data.index);
        idx = index[i];
        #print(idx);
        row = data.loc[idx,atts];

        #create the instance
        it = RowToInstance(row);

        # add instance to the list!
        I.append(it);

    return I;
#-----------------------------------------------------------------------

# this method select a random row from the data (dataframe) and return it,
# data : the dataset.
# atts : the set of attributes (the data columns).
# L : if its it None : so it will contains the select element index
#--------------------------------------------------------------
def GetRandomRow(data,atts, L):

    # initial an
    l=np.arange(len(data));
    l=list(l);

    # remove L element from l.
    if(L != None):
        l=sm.Remove(l,L);   
                                
    # random position.
    r = np.random.randint(0,len(data)-1);

    # get row :
    index = list(data.index);
    if(len(index)<20):
        print(index);
    
    # get the real value of index row !
    r = index[r];
    
    row = data.loc[r,atts];

    if(L != None):
        L.append(r);
        return (row,L);
    
    return row;
#------------------------------------------------------

# this method , allow as to generate random dataset from an original data.
# data : is the original dataset.
# size : the number of instances.
# withReplace : if true , so we allow process to get element more than one time

#-----------------------------------------------------
def GetRandomData(data, size, withReplace=True):
    # get attributes.
    atts = data.columns;
    L=None;
    if(not withReplace):
        L=list([]);
        
    # the data sample
    dat = pd.DataFrame(columns=atts);
    for i in range(0,size):
        row="";
        if(withReplace):
            row = GetRandomRow(data,atts,L);
        else:
            (row,L) = GetRandomRow(data,atts,L);
        #print(sm.toStr(row, atts));        
        dat = dat.append(row);

    print("data out ! ",len(dat));
    
    return dat;
#===========================================================
# Update 21 / 12

# this method take a data (dataframe) and generate set of boostrap data
# data : the main data.
# n : the number of sample to generate.
# size : the size of each sample
# inst : if true , return sample as instance (check instance file)
#------------------------------------------------------
def Boostrap(data,n=2,size=10,inst=False,withReplace=True):
    D = dict();
    
    # for each sample
    for i in range(0,n):
        # generate a random sample data.
        dat = GetRandomData(data, size,withReplace);
        print(dat);
        if (inst):
            # convert to Instance list
            dat = DataToInstances(dat,None,None);
            # put the dat in the dictionary D.
        D[str(i)]=dat;                
    return D;
#------------------------------------------------------

# this method show a simple example of the generation of random data from
# an origin one (the Boostrap process)
# data : the dataset !
#------------------------------------------------------
def BoostrapProcessExample(data):

    it = False;
    # D : is dictionary of (dataframe / list inst) depend on user param.
    D = Boostrap(data,n=2,size=10,inst=it,withReplace=True);
 
    # convert to instances
    if(it):
        I=D['1'];
    else:
        I = DataToInstances(D['1'],None,None);
    sm.Print(I,"instance");    
    exit();
    
    return;
#------------------------------------------------------    

#this method replace the NaN values , for now we replace with :
#   nominal values : with mode of the volumn values.
#   numeric values : with mean of the volumn values.
# data : the dataset that we want to update their values
# dTy : dictionary contains the set attributes type (nominal, numerical).
# Return ; the updated data.
#-----------------------------------------------------------------
def ReplaceNaNValues(data,dTy):
    # fillna not work well, check again or do my own format
    typ ='nominal'; T = GetSelect(dTy,typ);
    # for numeric variables replace nan by mean.
    for att in T:
        if att not in data.columns :
            continue;

        # get the mode(s) of the column
        mode = list(data[att].mode());
        # get element , we have to be sure that data is not empty !
        mode = mode[0];
        #update the column
        data[att] = data[att].fillna(mode);
    
    typ ='numerical'; T = GetSelect(dTy,typ);
    # for nominal variables replace nan by mode.
    for att in T:
        if att not in data.columns :
            continue;

        # get the column mean value.
        mean = data[att].mean();
        #update the column.
        data[att] = data[att].fillna(mean);

    return data;
#-----------------------------------------------------------------

# in this method we apply the Random Forest process.
# data : the input data (train data)
# test : the input data (test data)
# dTy : the attribut types (nominal , numerical)
# nbrTrees : the count of trees in forest
# maxDept : the maximum depth for the tree.
#-----------------------------------------
def RandomForest(data,test, dTy, clas, nbrTrees=4,maxDept = 3):

##    Proto-type
##    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
##            max_depth=2, max_features='auto', max_leaf_nodes=None,
##            min_impurity_decrease=0.0, min_impurity_split=None,
##            min_samples_leaf=1, min_samples_split=2,
##            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
##            oob_score=False, random_state=0, verbose=0, warm_start=False);

    # get target
    # replace the nan values by (mode , mean see method).
    data = ReplaceNaNValues(data,dTy);    
    test = ReplaceNaNValues(test,dTy);

    # transform data to labels (to be able to use it for training)
    data = TransformData(data);
    test = TransformData(test);  

    # get the target of train data
    T = data[clas];
    
    # remove the target from main data
    X = data.drop(columns=[clas]);
    
    # generate the classifier
    clf = RandomForestClassifier(n_estimators=nbrTrees, max_depth=maxDept,
                                 random_state=0);

    # training
    clf.fit(X, T);

    # generate the regressor
    regr = RandomForestRegressor(n_estimators=nbrTrees, max_depth=maxDept,
                                 random_state=0);
        
    regr.fit(X, T);
    
    # some print (feature importances_), 
    print(clf.feature_importances_);
    print(regr.feature_importances_);
 
    # predict class/ regression for test set !
    print(clf.predict(test));
    print(regr.predict(test));

    # we use kfold , it use for cross_val_score method
    kfold = 3;
    # compute the score of cross validation
    scores = cross_val_score(clf, X, T, cv=kfold);
    me = scores.mean();
    
    print(scores);print(me);

    return;
#-----------------------------------------------------





#----------- Main -------------------------------------
# -- Read input File ----------
filePath="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/Train.csv";
typeFile='csv';
data = ReadFile(filePath, typeFile);

# ---- Print Data ---
#PrintByRow(data,0,10,"\t");
I = DataToInstances(data,0,6);
# print instance !
sm.Print(I,"instance");

# --- do boostrap here -------------------------------
#BoostrapProcessExample(data);
#-----------------------------------------------------

# --- remove nan records -----------------------------
data = data.dropna();
#-----------------------------------------------------

#---- attributs type ---------------------------------
h = GetHeaderTypes(data[0:1]); print(h);
dTy=TypeHeader(h,data,False); print(dTy)
#-----------------------------------------------------

# ---- get Nominal / Numeric Data --------------------
typ='nominal'; dNom = SelectData(data,dTy,typ,False);
typ='numerical'; dNum = SelectData(data,dTy,typ,False);
#print(dNom); print(dNum);
#----------------------------------------------------

#------------ Some statistics : ----------------------
path="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/output.txt";
#DoStats(dNom , dTy, None,'_',path);
#DoStats(dNum , dTy, 4,'_',path);


#------------ generate Box plot for each attribut : -------------
picFolder = "D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/img";
#BoxPlotAllAtts(dNum,picFolder);

#--------------- K-means ----------------------------------------
# remove nan values to user Kmean
dNum = dNum.dropna();
#KmeansAlgorithm(dNum, True);

#---------------- Nomalize data (convert numeric data to nominal ones)
# remove the 'Outlet_Establishment_Year'
dNum =dNum.drop(columns=['Outlet_Establishment_Year']);

#dNum to dNom;
dNumToNom = NumericToNominal(dNum,4,'_');

# join two datasets !
dNom = JoinByColumns(dNom,dNumToNom);
dNom=dNom.dropna();

## --- apply A priory
#AprioriAlgorithm(dNom,True);

# --------- Decision Tree ---------------------

#re-load training data
typeFile='csv';
clas = 'Outlet_Type';

filePath="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/Train.csv";
dTrain = ReadFile(filePath, typeFile);

# --- test data

filePath="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/Test.csv";
dTest = ReadFile(filePath, typeFile);

#call our algo decision trees
pathOutGraph="D:/EdxCourses/Data Science/DataBases/bigmart-sales-data/graph_f.vz";
#DecisionTree(dTrain, dTest, clas, pathOutGraph);


# call process of Random Forest :
# the train data;
IN = dTrain;

# the test data.
TEST = dTest;
# generate the random forest.
RandomForest(IN,TEST, dTy, clas, 5, 4)



# --------- End of file --------------------------------







