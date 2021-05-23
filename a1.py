#created and edited by Samuel Phillips

#imports for data, classes and more
from pandas import DataFrame
import pandas as pd
import numpy as np
import math
import d_similarity_measure
import csv
from sklearn.datasets import load_iris

#function to get either euclidean or cosine distance from two vars
#a1p1
def getDist(mType, a1, a2):
    if mType == 'Euclidean' or mType == 'euclidean':
        object_get_similarity = d_similarity_measure.simimarity_measure(a1, a2)
        euclidean_distance = object_get_similarity.get_euclidean()
        x = euclidean_distance
    elif mType == 'Cosine' or mType == 'cosine':
        object_get_similarity = d_similarity_measure.simimarity_measure(a1, a2)
        cosine_distance = object_get_similarity.get_cosine()
        x = cosine_distance
    return x

#function that splits a long list into segments for showing better data
#a1p1
def splitList(l,n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

#function that gets the Gini from two parameters
#a1p2
def getGini(x,y):
    denom = float(x + y)
    p1 = float(x/denom)
    p2 = float(y/denom)
    gini = float(p1**2 + p2**2)
    gini = float(round(1 - gini,2))
    return gini

#function that gets the Entropy from two parameters
#a1p2
def getEntropy(x,y):
    denom = x + y
    p1 = float(x/denom)
    p2 = float(y/denom)
    e = float(round(-(p1 * math.log(p1,2)) - (p2 * math.log(p2,2)),2))
    return e

#function that gets the Maxclassification Error from two parameters
#a1p2
def getME(x,y):
    denom = float(x + y)
    p1 = float(x/denom)
    
    p2 = float(y/denom)
    me = float(round(1 - np.max([p1,p2]),2))
    return me

#-- a1p1 starts here --
#iris data is loaded
iris = load_iris()
iris_data = iris.data

#data lists are created
e = []
c = []

#nested for loop that gets the cosine and euclidean distance 
#between every two plants in the iris data
for i in range(0, len(iris_data)):
    for q in range(0, len(iris_data)):
        c.append(getDist('Cosine', iris_data[i], iris_data[q]))
        e.append(getDist('Euclidean', iris_data[i], iris_data[q]))

#long list of cosine distances is split into 150 lists of 150 distances
x = list(splitList(c, 150))

#long list of cosine distances is split into 150 lists of 150 distances
y = list(splitList(e, 150))

#data frame for the cosine distance data is created
cos = DataFrame(x)
#convert dataframe into csv file
cos.to_csv('iris_cosine.csv') 

#data frame for the cosine distance data is created
euc = DataFrame(y)
#convert dataframe into csv file
euc.to_csv('iris_euclidean.csv')
#-- a1p1 ends here --

#-- a1p2 starts here --
#neccessary vars, lists, dicts and more are declared
g, e, m, data, d2, mkrs, p1, p2 = [], [], [], {}, {}, [], [], []
gGain, eGain, mGain = [], [], []

#data frame of all the necessary info is made
tData = {'ID':[1,2,3,4,5,6,7,8,9,10],
        'Home Owner':['Yes','No','No','Yes','No','No','Yes','No','No', 'No'],
        'Marital Status':['Single', 'Married', 'Single', 'Married', 'Divorced', 'Married', 'Divorced', 'Single', 'Married', 'Single'],
        'Annual Income':[125000, 100000, 70000, 120000, 95000, 60000, 220000, 85000, 75000, 90000],
        'Defaulted Borrower':['No','No','No','No','Yes','No','No','Yes','No','Yes']}
df = DataFrame(tData)

#code that gets the gini, ME and entropy from the home owner node
#also removes unneccessary rows from the dataframe
n1, n2, c, x, mkrs = 0, 0, 0, [], []
x = df['Home Owner'].to_list()
for i in range(0,len(x)):
    if x[i] == 'Yes':
        n1 += 1
        mkrs.append(i)
    else:
        n2 += 1 
g.append(getGini(n1,n2))
e.append(getEntropy(n1,n2))
m.append(getME(n1,n2))
p1.append(n2)
p2.append(n1)
for i in range(0,len(mkrs)):
    df.drop(index=mkrs[i], inplace=True)
df = df.reset_index(drop=True)
  
#code that gets the gini, ME and entropy from the marital status node
#also removes unneccessary rows from the dataframe
n1, n2, c, x, mkrs = 0, 0, 0, [], []
x = df['Marital Status'].to_list()
for i in range(0,len(x)):
    if x[i] == 'Married':
        n1 += 1
        mkrs.append(i)
    else:
        n2 += 1   
g.append(getGini(n1,n2))
e.append(getEntropy(n1,n2))
m.append(getME(n1,n2))
p1.append(n2)
p2.append(n1)
for i in range(0,len(mkrs)):
    df.drop(index=mkrs[i], inplace=True)
df = df.reset_index(drop=True)
    
#code that gets the gini, ME and entropy from the annual income node
#also removes unneccessary rows from the dataframe
n1, n2, c, x, mkrs = 0, 0, 0, [], []
x = df['Annual Income'].to_list()
for i in range(0,len(x)):
    if x[i] > 80000:
        n1 += 1
        mkrs.append(i)
    else:
        n2 += 1   
g.append(getGini(n1,n2))
e.append(1-getEntropy(n1,n2))
m.append(getME(n1,n2))
p1.append(n1)
p2.append(n2)
for i in range(0,len(mkrs)):
    df.drop(index=mkrs[i], inplace=True)
df = df.reset_index(drop=True)

#for loop that calcuates the gini gain for each major node
for i in range(0, len(g)):
    if i < 2:
        den = p2[i] + p1[i]
        prop = p1[i] / den
        x = g[i] - 0 - prop * g[i+1]
        gGain.append(round(x,2))
    else:
        x = round(g[i] - 0 - 0,2)
        gGain.append(x)

#for loop that calcuates the entropy gain for each major node
for i in range(0, len(e)):
    if i < 2:
        den = p2[i] + p1[i]
        prop = p1[i] / den
        x = e[i] - 0 - prop * e[i+1]
        eGain.append(round(x,2))
    else:
        x = round(e[i] - 0 - 0,2)
        eGain.append(x)

#for loop that calcuates the ME gain for each major node
for i in range(0, len(m)):
    if i < 2:
        den = p2[i] + p1[i]
        prop = p1[i] / den
        x = m[i] - 0 - prop * m[i+1]
        mGain.append(round(x,2))
    else:
        x = round(m[i] - 0 - 0,2)
        mGain.append(x)

#data is made into a dictionary
data = {'Gini' : g, 'Entropy': e, 'Maxclassification Error': m}
d2 = {'Gini' : gGain, 'Entropy': eGain, 'Maxclassification Error': mGain}

#data dictionary is made into a dataframe
secDF = DataFrame(data)
thirDF = DataFrame(d2)

#indexes for the dataframe are edited to better showcase the data
secDF.index = ['Home Owner', 'Marital Status', 'Annual Income']
thirDF.index = ['Home Owner', 'Marital Status', 'Annual Income']
secDF.header = 'Impurity'
thirDF.header = 'Gain'

#dataframe is outputted to a xlsx or excel file
writer = pd.ExcelWriter('dtree.xlsx', engine='xlsxwriter')
secDF.to_excel(writer, sheet_name='Sheet1', startrow=1)
thirDF.to_excel(writer, sheet_name='Sheet1', startrow=7)
s = writer.sheets['Sheet1']
s.write_string(0,0, 'Impurity')
s.write_string(6,0, 'Gain')
writer.save()
#-- a1p2 ends here --