#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:58:50 2023

data project 1

@author: lynnmiao
"""
"""
1) Are movies that are more popular (operationalized as having more ratings) rated higher than movies that
are less popular? [Hint: You can do a median-split of popularity to determine high vs. low popularity movies]
- divide the list into two "more" vs. "less" popular
- find the avg ratings for two group
- independent t-test? welch test?

2) Are movies that are newer rated differently than movies that are older? [Hint: Do a median split of year of
release to contrast movies in terms of whether they are old or new]
- divide the list into two older vs. newer
- find the avg ratings for two group
- independent t-test? welch test?


3) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
- split reviews of shrek by gender
- find avg ratings
- independent t test

4) What proportion of movies are rated differently by male and female viewers?
?

5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
- split reviews of lion king into only children vs. siblings
- find avg ratings
- independent t test

6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings
vs. those without?
?

7) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who
prefer to watch them alone?
- split reviews of lion king into only children vs. siblings
- find avg ratings
- independent t test

8) What proportion of movies exhibit such a “social watching” effect?


9) Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?
KS test

10) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana
Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. How many of these
are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks
featured in this question to identify the movies that are part of each franchise]
ANOVA?


Extra Credit: Tell us something interesting and true (supported by a significance test of some kind) about the
movies in this dataset that is not already covered by the questions above [for 5% of the grade score].

"""


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

