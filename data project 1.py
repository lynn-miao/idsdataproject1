#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:58:50 2023

data project 1

@author: lynnmiao
"""

import sqlite3
import random
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, permutation_test
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #adds a style to seaborn plots
import scipy.stats as stats
import re


#%%
# Read CSV file
df = pd.read_csv('movieReplicationSet.csv') # ingest the data in one line

# Pull out movie ratings only
ratings = df.apply(pd.to_numeric, errors='coerce') # makes sure all strings are converted to floats

ratings = ratings.iloc[:, :400]
#%%


#%%

"""
1) Are movies that are more popular (operationalized as having more ratings) rated higher than movies that
are less popular? [Hint: You can do a median-split of popularity to determine high vs. low popularity movies]
"""

# find the median number of ratings per movie
count = ratings.count(axis=0, numeric_only=True)
median_ratings_ct = count.median()

# create a df for popular and unpopular
top_half = count > median_ratings_ct
bottom_half = count < median_ratings_ct

popular_df = ratings.loc[:, top_half]
unpopular_df = ratings.loc[:, bottom_half]
 
# find the avg ratings for popular movies and unpopular movies
popular_ratings = popular_df.mean()
unpopular_ratings = unpopular_df.mean()

popular_mean = popular_ratings.mean()
unpopular_mean = unpopular_ratings.mean()

# check the variance of both groups
var_pop = np.var(popular_ratings)
var_unpop = np.var(unpopular_ratings)

# Assuming you have two groups of data: group1 and group2
t_stat, p_value = stats.ttest_ind(popular_ratings, unpopular_ratings)

# Display the t-statistic and p-value
print("t-statistic:", t_stat)
print("p-value:", p_value)
#t-statistic: 17.7560492698737
#p-value: 2.2696530276564846e-52
 #%%
 


#%%
"""
2) Are movies that are newer rated differently than movies that are older? [Hint: Do a median split of year of
release to contrast movies in terms of whether they are old or new]
"""

pattern = r'\((\d{4})\)'

def extract_year_from_column_name(column_name):
    match = re.search(pattern, column_name)
    if match:
        return match.group(1)
    else:
        return None

extracted_years = ratings.columns.to_series().apply(extract_year_from_column_name)

median_year = extracted_years.median()

newer = extracted_years[0:].astype(int) >= median_year
older = extracted_years[0:].astype(int) < median_year

newer_ratings = ratings.loc[:, newer]
older_ratings = ratings.loc[:, older]

newer_ratings_avg = newer_ratings.mean()
older_ratings_avg = older_ratings.mean()

avg_newer = newer_ratings_avg.mean()
avg_older = older_ratings_avg.mean()


t_stat, p_value = stats.ttest_ind(newer_ratings_avg, older_ratings_avg)
ks_statistic, ks_p_value = stats.kstest(newer_ratings_avg, older_ratings_avg)


print("t-statistic:", t_stat)
print("p-value:", p_value)

print("ks statistic:", ks_statistic)
print("ks p_value:", ks_p_value)

#t-statistic: 1.605479609469478
#p-value: 0.10918141397982746
#ks statistic: 0.12087719736940812
#ks p_value: 0.09607890866065033

#%%


#%%
"""
3) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
"""

# create Shrek df
column_number = df.columns.get_loc('Shrek (2001)')

Shrek = df.iloc[:,[column_number,474]]

# pull out null values on shrek rating or gender

Shrek = Shrek.dropna()

# split into Male/Female

female = Shrek[Shrek.iloc[:,1] == 1]
male = Shrek[Shrek.iloc[:,1] == 2]

t_stat, p_value = stats.ttest_ind(male.iloc[:,0], female.iloc[:,0],equal_var=False)
ks_statistic, ks_p_value = stats.kstest(male.iloc[:,0], female.iloc[:,0])

print("t-statistic:", t_stat)
print("p-value:", p_value)

print("ks statistic:", ks_statistic)
print("ks p_value:", ks_p_value)

#t-statistic: -1.1558907155973421
#p-value: 0.24834907946281018
#ks statistic: 0.09796552051512596
#ks p_value: 0.05608204072286342
#%%


#%%
"""
4) What proportion of movies are rated differently by male and female viewers?
"""
gender = df.iloc[:,list(range(400)) + [474]]

female_ratings = gender[gender.iloc[:,400] == 1]
male_ratings = gender[gender.iloc[:,400] == 2]

# initialize list of movies where p-value < .005
p = .005
different = []

for i in range(400):
    t_stat, p_value = stats.ttest_ind(female_ratings.iloc[:,i].dropna(), male_ratings.iloc[:,i].dropna()
                                      ,equal_var=False)
    if p_value < p:
        different.append(female_ratings.columns[i])
        
# initialize list of movies where p-value < .005
p = .005
different_ks = []

for i in range(400):
    t_stat, p_value = stats.kstest(female_ratings.iloc[:,i].dropna(), male_ratings.iloc[:,i].dropna())
    if p_value < p:
        different_ks.append(female_ratings.columns[i])

common_elements = set(different).intersection(different_ks)
number_of_common_elements = len(common_elements)
print("Number of common elements:", number_of_common_elements)


#%%

"""
5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
"""
#%%

# create df
column_number = df.columns.get_loc('The Lion King (1994)')

lk = df.iloc[:,[column_number,475]]

# pull out null values on rating or gender

lk = lk.dropna()

# split into only child status

only_yes = lk[lk.iloc[:,1] == 1]
only_no = lk[lk.iloc[:,1] == 0]

t_stat, p_value = stats.ttest_ind(only_yes.iloc[:,0], only_no.iloc[:,0],equal_var=False)

print("t-statistic:", t_stat)
print("p-value:", p_value)

#t-statistic: -1.884028409511613
#p-value: 0.06102886373552747
#%%



#%%

"""
6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings
vs. those without?
?
"""

only = df.iloc[:,list(range(400)) + [475]]

only_ratings = only[only.iloc[:,400] == 1]
not_only_ratings = only[only.iloc[:,400] == 0]

# initialize list of movies where p-value < .005
p = .005

only_different = []

for i in range(400):
    t_stat, p_value = stats.ttest_ind(only_ratings.iloc[:,i].dropna()
                                      , not_only_ratings.iloc[:,i].dropna()
                                      , equal_var=False)
    if p_value < p:
        only_different.append(not_only_ratings.columns[i])
        
only_different_ks = []

for i in range(400):
    t_stat, p_value = stats.kstest(only_ratings.iloc[:,i].dropna()
                                      , not_only_ratings.iloc[:,i].dropna())
    if p_value < p:
        only_different.append(not_only_ratings.columns[i])

#%%



#%%
"""
7) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who
prefer to watch them alone?

"""

# create df
column_number = df.columns.get_loc('The Wolf of Wall Street (2013)')

wws = df.iloc[:,[column_number,476]]

# pull out null values on rating or gender

wws = wws.dropna()

# split 

alone = wws[wws.iloc[:,1] == 1]
alone_no = wws[wws.iloc[:,1] == 0]

t_stat, p_value = stats.ttest_ind(alone.iloc[:,0], alone_no.iloc[:,0]
                                  ,equal_var=False)

print("t-statistic:", t_stat)
print("p-value:", p_value)

#t-statistic: 1.5513309472217705
#p-value: 0.12139103950020742
#%%

#%%
"""
8) What proportion of movies exhibit such a “social watching” effect?
"""
social = df.iloc[:,list(range(400)) + [476]]

social_ratings = social[social.iloc[:,400] == 1]
social_not_ratings = social[social.iloc[:,400] == 0]

# initialize list of movies where p-value < .005
p = .005

social_different = []

for i in range(400):
    t_stat, p_value = stats.ttest_ind(social_ratings.iloc[:,i].dropna()
                                      , social_not_ratings.iloc[:,i].dropna()
                                      , equal_var=False)
    if p_value < p:
        social_different.append(social_ratings.columns[i])
        
#%%


#%%

"""
9) Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?
KS test
"""

home = ratings['Home Alone (1990)'].dropna()
nemo = ratings['Finding Nemo (2003)'].dropna()

ks_statistic, p_value = stats.kstest(home, nemo)

print("ks_stat:",ks_statistic)
print("p-value:", p_value)

#ks_stat: 0.15269080020897632
#p-value: 6.379397182836346e-10

#%%

"""
10) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana
Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. How many of these
are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks
featured in this question to identify the movies that are part of each franchise]
ANOVA?
"""
#%%

series = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 'Jurassic Park', 
          'Pirates of the Caribbean', 'Toy Story', 'Batman']
results = []

titles = df.columns 

title = 'Star Wars' # or any other title, for that matter
Star_Wars = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(Star_Wars.iloc[:,0].dropna()
                     ,Star_Wars.iloc[:,1].dropna()
                     ,Star_Wars.iloc[:,2].dropna()
                     ,Star_Wars.iloc[:,3].dropna()
                     ,Star_Wars.iloc[:,4].dropna()
                     ,Star_Wars.iloc[:,5].dropna())

results.append([title,p])

#%%
#%%
title = 'Harry Potter' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna()
                     ,movie.iloc[:,3].dropna())

results.append([title,p])

#%%

title = 'The Matrix' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna())
results.append([title,p])

#%%
#%%

title = 'Indiana Jones' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna()
                     ,movie.iloc[:,3].dropna())
results.append([title,p])

#%%
#%%

title = 'Jurassic Park' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna())
results.append([title,p])

#%%
#%%

title = 'Pirates of the Caribbean' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna())
results.append([title,p])

#%%
#%%

title = 'Toy Story' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna())
results.append([title,p])

#%%
#%%

title = 'Batman' # or any other title, for that matter
movie = df.loc[:,df.columns.str.contains(title)]
f,p = stats.f_oneway(movie.iloc[:,0].dropna()
                     ,movie.iloc[:,1].dropna()
                     ,movie.iloc[:,2].dropna())
results.append([title,p])

#%%

#%%
"""
Extra Credit: Tell us something interesting and true (supported by a significance test of some kind) about the
movies in this dataset that is not already covered by the questions above [for 5% of the grade score].
"""

# Do people who are movie buffs (view more than 200 movies) rate movies differently than everyone else?

row_non_null_count = ratings.notnull().sum(axis=1)

# Filter and keep rows with 20 or fewer non-null values
threshold = 200
ratings_buff = df[row_non_null_count >= threshold]
ratings_non_buff = df[row_non_null_count < threshold]

buff_avg = ratings_buff.mean()
buff_non_avg = ratings_non_buff.mean()
buff_mean_rating = buff_avg.mean()
buff_non_mean_rating = buff_non_avg.mean()


t_stat, p_value = stats.ttest_ind(buff_avg, buff_non_avg, equal_var=False)

print("Avg buff rating:", buff_mean_rating)
print("Avg non-buff rating:", buff_non_mean_rating)
print("t-statistic:", t_stat)
print("p-value:", p_value)
#%%