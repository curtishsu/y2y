
# coding: utf-8

# In[139]:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score 
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.svm.libsvm import predict_proba
get_ipython().magic(u'matplotlib inline')


# In[290]:

#Import consequences
consequences = pd.read_csv('datasets/Consequence_datamine.csv')
consequences = consequences.iloc[4:,:]
consequences['Discipline'] = consequences['Discipline'].str.title()
consequences['Type of Incident'] = consequences['Type of Incident'].str.title()


# In[7]:

#Create a function that inputs dataframe
#Output is a dataframe 
def quick_summary(consequences):
    types_warnings = consequences['Type of Incident'].unique()
    types_incident = (consequences['Discipline'].unique())
    guest_warnings = pd.DataFrame(consequences['Guest Name'].value_counts())
    guest_discipline = pd.DataFrame(consequences['Guest Name'].value_counts())
    guest_warnings.columns = ['Total Warnings']
    guest_discipline.columns = ['Total Warnings']
    for typ in types_warnings:
        guest_warnings_typ = pd.DataFrame(consequences['Guest Name'][consequences['Type of Incident'] == typ].value_counts())
        guest_warnings_typ.columns = [typ]
        guest_warnings = pd.concat([guest_warnings, guest_warnings_typ], axis=1)
    
    for ty in types_incident:
        guest_warnings_ty = pd.DataFrame(consequences['Guest Name'][consequences['Discipline'] == ty].value_counts())
        guest_warnings_ty.columns = [ty]
        guest_discipline = pd.concat([guest_discipline, guest_warnings_ty], axis = 1)
        
    guest_warnings = guest_warnings.fillna(0)
    guest_warnings = guest_warnings.sort_values('Total Warnings', ascending = False)
    guest_discipline = guest_discipline.fillna(0)
    guest_discipline = guest_discipline.sort_values('Total Warnings', ascending = False)
    
    return guest_warnings, guest_discipline


# In[20]:

#run quick sumary and write to excel file
warnings, discipline = quick_summary(consequences)
warnings = warnings.drop(warnings.columns[[5]], axis = 1)
writer = pd.ExcelWriter('consequences.xlsx')
warnings.to_excel(writer,'Guest Type of Indicent')

# data.fillna() or similar.
discipline.to_excel(writer,'Guest - Discipline')
writer.save()


# In[285]:

#Create a function that will read the descriptions and predict the type of indicent
descriptions = consequences['Description'].values
types_incident = consequences['Type of Incident'].values
df = pd.DataFrame({'descriptions':descriptions, 'types':types_incident})
df = df.dropna()
descriptions = df['descriptions'].values
types_incident = df['types']


#Vectorize
vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the'], min_df=1)
corpus_x = descriptions
x = vectorizer.fit_transform(corpus_x)
x = x.toarray()


# In[216]:

#Generate a dictionary with a value for each type of warning, then attribute this to each value
all_types = df['types'].unique()
type_vals = {}
i = 0
for t in all_types:
    type_vals[t] = i
    i +=1
types_incident = types_incident.values
y = []
for t in types_incident:
    y.append(type_vals.get(t))


# In[222]:

#Split data into train and test
#Predict using LDA
xtrain, xtest, ytrain, ytest = sk_split(x, y,  train_size = .7)
lda = da.LinearDiscriminantAnalysis()
lda.fit(xtrain, ytrain)
preds = lda.predict(xtest)
score = lda.score(xtest, ytest)


# In[264]:

#Create a function that takes the warnings and breaks it up into a list of discipline
def warning_date_generator(consequences):
    dates = consequences['Date Issued'].values
    discipline = consequences['Discipline'].values
    disc_dates_lst = []
    for i in range(0, len(discipline)):
        dt_lst = dates[i].split('/')
        dt_lst = map(int, dt_lst)
        dt_lst.append(discipline[i])
        disc_dates_lst.append(dt_lst)
    return disc_dates_lst


# In[312]:

#Generate the times for the weeks
import time
import datetime
from datetime import date
start_date = date(2016, 1, 1)
dates = []
dates.append(start_date)
for i in range (0,15):
    start_date =  start_date + datetime.timedelta(days = 7)
    dates.append(start_date)


# In[292]:

#create a function that maps number of certain types of incidence and warnings
#Find the number of warnings and types of incident on a weekly basis 
#Loop through all of consequences data frame once. 
#Create lists for each of the different types of consequences
#To account for weeks, create a variable for the current date
#if the day is before 21, then we can treat the end date of current from as start date + 7
#If the day is after 21, then figure out how many days until the end of the week and set that as the end date. 

def warning_plots(consequences):
    from calendar import monthrange
    discipline_types = consequences['Discipline'].unique()
    disciplines = {}
    disc_count = {}
    for d_type in discipline_types:
        disciplines[d_type] = [0]
        disc_count[d_type] = 0
    disc_dates_lst = warning_date_generator(consequences)
    start_date = 1
    start_month = 1
    start_year = 2016
    month_end = monthrange(start_year, start_month)
    month_end = month_end[1]
    i = 0
    for d in disc_dates_lst:
        disc = d[3]
        mon = d[0]
        date = d[1]
        yr = d[2]
        
        if start_date + 7 < month_end:
            if date < start_date + 7:
                disc_count[disc] += 1
            elif date >= start_date + 7:
                for d_type in discipline_types:
                    disciplines[d_type].append(disc_count[d_type])
                    disc_count[d_type] = 0
                disc_count[disc] +=1
                start_date = start_date + 7
                
        else:
            if (mon <> start_month):
                date = date + month_end
                
            if date < start_date + 7:
                disc_count[disc] += 1
            elif date >= start_date + 7:
                for d_type in discipline_types:
                    disciplines[d_type].append(disc_count[d_type])
                    disc_count[d_type] = 0
                disc_count[disc] +=1
                start_date = start_date + 7
                start_date = start_date%month_end
                start_month = mon
                month_end = monthrange(start_year, start_month)
                month_end = month_end[1]
    return disciplines
                


# In[305]:

discipline_breakdowns = warning_plots(consequences)


# In[316]:

print len(dates)
print len(discipline_breakdowns)


# In[323]:

fig, ax = plt.subplots(5, 1, figsize = (10,25))
discipline_types = consequences['Discipline'].unique()
i = 0
for d in discipline_types:
    ax[i].plot(dates, discipline_breakdowns[d])
    ax[i].set_ylabel('Number of Warnings')
    ax[i].set_title(d)
    i +=1
    


# In[ ]:



