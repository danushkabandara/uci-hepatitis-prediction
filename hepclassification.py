import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
attribute_names = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malais","Anorexia", "Liver_big","Liver_firm","Spleen_palpable","Spiders","Ascites", "Varices","Bilirubin","Alk_Phosphate","SGOT", "Albumin","Protime","Histology"]
df = pd.read_csv(url,names=attribute_names)
df = df.applymap(str)#convert df to str to access '?' as str
for column in df:
    counts = df[column].value_counts().to_dict() #get counts of each unique value to a map
    countOfQMarks = counts.get("?",0)
    percentMissingValues = (100*countOfQMarks/ (df[column].count()-countOfQMarks))
    print "Column: %-16s Percentage of Missing Values: %d"%(column,  percentMissingValues)+"%"    
    
df = df.replace('?', np.nan)#missing value replacement
df = df.apply(pd.to_numeric)#convert back to numbers for ease of calculation
dfLive = df.loc[(df['Class'] == 1)] #separate the datapoints to Live, Die 
dfDead = df.loc[(df['Class'] == 2)]

cont_columns=["Age", "Bilirubin","Alk_Phosphate","SGOT", "Albumin","Protime"]
bool_columns = ["Sex", "Steroid", "Antivirals", "Fatigue", "Malais","Anorexia", "Liver_big","Liver_firm","Spleen_palpable","Spiders","Ascites", "Varices","Histology"]

means_dfLive_bool = []
means_dfDead_bool = []
means_dfLive_cont = []
means_dfDead_cont = []
for column in dfLive:
    if column == "Class":
        continue
    if column in cont_columns:
        means_dfLive_cont.append(dfLive[column].mean()) #calculate means for visualization
    else:
        means_dfLive_bool.append(dfLive[column].mean())
for column in dfDead:
    if column == "Class":
        continue
    if column in cont_columns:
        means_dfDead_cont.append(dfDead[column].mean())  
    else:
        means_dfDead_bool.append(dfDead[column].mean())  

#############################creating plot for boolean values
fig, ax = plt.subplots(figsize=(15,5))
N=13
ind = np.arange(N)    # the x locations for the groups       
p1 = ax.bar(ind, means_dfLive_bool, color = 'r',width = 0.2)
p2 = ax.bar(ind + 0.2, means_dfDead_bool, color = 'y',width = 0.2)
ax.set_title('Mean of each boolean attribute by Class Value')
ax.legend((p1[0], p2[0]), ('Live', 'Die'))
ax.autoscale_view()
plt.xticks(range(0,13),bool_columns)
plt.show()
                       
###########################creating plot for non boolean values
fig, ax = plt.subplots(figsize = (15,5))
N=6
ind = np.arange(N)    # the x locations for the groups       
p1 = ax.bar(ind, means_dfLive_cont, color = 'r', width = 0.2)
p2 = ax.bar(ind + 0.2, means_dfDead_cont, color='y', width = 0.2)
ax.set_title('Mean of each non boolean attribute by Class Value')
ax.legend((p1[0], p2[0]), ('Live', 'Die'))
ax.autoscale_view()
plt.xticks(range(0,6),cont_columns)
plt.show()

############################creating heatmap of pairwise correlations
import seaborn as sb
fig, ax = plt.subplots(figsize=(15,15)) 
ax.set_title('Heatmap of the pairwise correlation of attribute columns')
sb.heatmap(df.corr(), annot = True, linewidths = .5, ax = ax)

#############################simple statistics
corrDict = {}
for column in df:
    if column == "Class":
        continue
    else:
        corrDict[column] = abs(df['Class'].corr(df[column]))
sortedCorr = sorted(corrDict, key = corrDict.get, reverse = True)
print "Top 5 most correlated features:"
top5Corr = sortedCorr[:5]
for i in range(len(top5Corr)):
    print i+1, top5Corr[i]
    
##############################classification  
    
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
import itertools
df= df.dropna() #drop NaNs from dataframe
scaler = MinMaxScaler()
X = df.iloc[:,1:]
y = df['Class']
#use grid search to get the best value for C
model = LinearSVC()
params = {"C": np.arange(1, 25, 0.1)}
opt = GridSearchCV(model, params, cv = 5)
opt.fit(scaler.fit_transform(X), y)
#cross validation with optimized C
optimized_model=LinearSVC(C=opt.best_params_['C'])
fit = optimized_model.fit(scaler.fit_transform(X), y) #scale the data to between 0 and 1
scores = cross_val_score(optimized_model, scaler.fit_transform(X), y, cv = 5)
print "Accuracy from 5 fold xvalidation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

##################################get the most predictive features
coefficients = abs(fit.coef_)
temp = np.argsort(coefficients) #the order of indexes when sorting by increasing predictiveness
indeces_ascending_order= list(itertools.chain.from_iterable(temp))
print "Most predictive features in descending order:"
bulletNum = 1
for i in range(len(indeces_ascending_order)-1, -1, -1):#printing the features in descending order of predictiveness
    print bulletNum, attribute_names[indeces_ascending_order[i]] #attribute_names include class so add 1 to skip class
    bulletNum += 1    
