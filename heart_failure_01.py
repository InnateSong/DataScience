# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 00:04:56 2020

@author: Deivydas Vladimirovas
"""

import pandas as pd
import matplotlib.pyplot as plt
#some settings for pd (expands the output)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

def getInfo(df):
    print(list(df.columns))     #Returns the column names   
    print(df.dtypes)            #returns the data types of each column
    print(df.shape[0])          #Returns the number of rows
    print(df.shape[1])          #Returns the number of columns
    print(df.describe())        #Returns numeric decriptions of numeric columns
    print(df.isnull().sum())    #Returns table of missing values in columns
    
df = pd.read_csv("datasets_727551_1263738_heart_failure_clinical_records_dataset.csv")
getInfo(df)
print(df.head())
print(df["DEATH_EVENT"].values)
print("Heart Failure started from: ",df["age"].min())       #Returns the minimum value in the column
print("Heart Failure reached to age: ", df["age"].max())    ##Returns the maximum value in the column
#SMOKING RELATIONSHIP TO NUMBER OF DEATHS
smokers = df.loc[df["smoking"]==1, "smoking"].count()       #getting the value of a condition
Nsmokers = df.loc[df["smoking"]!=1, "smoking"].count()      #getting the value of a condition
#plt.bar(["name of y-axis"],data1, label)
plt.bar(["Number of smokers"], smokers, label="Smokers")    
#plt.bar(["name of y-axis"],data2,bottom=data1, label) [this creates a stacked bar chart]
plt.bar(["Number of smokers"],Nsmokers, bottom=smokers, label="Non smokers")
#plt.bar(["name of y-axis"],data3, label)
plt.bar(["Number of deaths from heart failure"], df.shape[0], label="Total number")
plt.ylabel("Number of deaths")
plt.legend()    #plots the labels on top right corner
plt.show()
#GENDER AND DEATHS
death_df = df.loc[df["DEATH_EVENT"] == 1]               
female=death_df.loc[death_df["sex"]==0,"sex"].count()   #getting the value of a condition
male=death_df.loc[death_df["sex"]==1,"sex"].count()     #getting the value of a condition
plt.bar(["Number of females"], female)
plt.bar(["Number of males"], male)
plt.ylabel("Number of deaths")
plt.show()
#Diabetic and deaths
DiabY = df.loc[df["diabetes"] == 1, "diabetes"].count()
DiabN = df.loc[df["diabetes"] == 0, "diabetes"].count()
plt.bar(["Diabetic"], DiabY, label="diabetic")
plt.bar(["Diabetic"], DiabN, bottom=DiabY, label="Not diabetic")
plt.ylabel("Number of Deaths")
plt.show()
#AGE and death
ages_df = df.sort_values(by="age", ascending=False, na_position="first")
ages = ages_df["age"].value_counts(sort=True, ascending=False, dropna=False)
print(ages.index.tolist())
print(list(ages))
plt.bar(ages.index.tolist(), list(ages))
plt.xlabel("ages ranging from 40-95")
plt.ylabel("number of deaths")
plt.show()
#High blood pressure and deaths
HighBYes = df.loc[df["high_blood_pressure"]==1, "high_blood_pressure"].count()
HighBNo = df.loc[df["high_blood_pressure"]==0, "high_blood_pressure"].count()
width = 0.5
plt.bar(["Blood pressure"],HighBYes, width=width,label='HighBloodPressure')
plt.bar(["Blood pressure"], HighBNo,width=width, label='NormalBloodPressure',  bottom=HighBYes)
plt.ylabel("Number of deaths")
plt.legend()
plt.show()
#USING sklearn 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = df[['platelets']]
Y = range(df.shape[0])
print(Y)
lm.fit(X, Y)
print(lm.intercept_)
print(lm.coef_)

import seaborn as sns
sns.regplot(x='age', y='platelets', data=df)
plt.ylim(0, )
plt.show()
plt.close()

#IS AGE AND SEX AN INDICATOR FOR DEATH EVENTS?
"""Distribution plot to show any relationship betweem age and daeaths"""
sns.distplot(death_df["age"], label="Ages ranging from 40 to 95",)
plt.legend()
plt.title("Distribution/Histogram showing relationship betwem the ages and number of deaths")
plt.savefig("Age_death_histogram.png")
plt.show()
plt.close()
"""Box plot to show any relationship between sex and age"""
ageM = df.loc[df["sex"]==1, "age"].values
ageF = df.loc[df["sex"]==0, "age"].values
box_plot = [ageM, ageF]
plt.boxplot(box_plot,vert=0, labels=["males", "females"])
plt.xlabel("Ages")
plt.title("Box plot graph showing the relationship between ages and sex")
plt.savefig("age_sex_boxplot.png")
plt.show()
plt.close()
"""Survival percentage for males and females"""
death_df = df.loc[df["DEATH_EVENT"] == 1]
femaleD=death_df.loc[death_df["sex"]==0,"sex"].count()
maleD=death_df.loc[death_df["sex"]==1,"sex"].count()
print("Number of females died: ", femaleD)
print("Number of males died: ", maleD)
Live_df = df.loc[df["DEATH_EVENT"] == 0]
femaleL=Live_df.loc[Live_df["sex"]==0,"sex"].count()
maleL=Live_df.loc[Live_df["sex"]==1,"sex"].count()
print("Number of females survived: ", femaleL)
print("Number of males survived: ", maleL)
labels = 'Female Deaths', 'Male deaths', 'Female survivors','male survivors'
sizes = [femaleD, maleD, femaleL, maleL]
explode=(0.1, 0.1, 0.1, 0.1)
plt.pie(sizes, explode =explode,labels=labels, autopct='%1.1f%%')
plt.axis("equal")
plt.savefig("sex_survival_pie_chart.png")
plt.show()
plt.close()
"""Analysis on age surival number"""
plt.subplots(figsize=(10,6), dpi=100)
sns.distplot(df.loc[df["DEATH_EVENT"]==1, "age"], color="dodgerblue", label="Died")
sns.distplot(df.loc[df["DEATH_EVENT"]==0, "age"], color="orange",label="Survived") 
plt.legend()
plt.savefig("age_survival_histogram.png")
plt.show()
plt.close()
"""analysis in age and smoking"""
death_df = df.loc[df["DEATH_EVENT"]==1]
live_df = df.loc[df["DEATH_EVENT"]==0]
fig, ax=plt.subplots()

live_ageS = live_df.loc[live_df["smoking"]==1, "age"].values
live_ageNS = live_df.loc[live_df["smoking"]==0, "age"].values
dead_ageS = death_df.loc[death_df["smoking"]==1, "age"].values
dead_ageNS = death_df.loc[death_df["smoking"]==0, "age"].values

live_ages = [live_ageS, live_ageNS]
dead_ages = [dead_ageS, dead_ageNS]
bp1=ax.boxplot(live_ages, positions=[1,4],  widths = 0.35, patch_artist=True)
bp2=ax.boxplot(dead_ages, positions=[2,5],  widths = 0.35, patch_artist=True)
ax.set_xticklabels(['Smokers', 'Non Smokers'])
ax.set_xticks([1.5,4.5])

c1="red"
c2="blue"
for item in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bp1[item], color=c1)
        plt.setp(bp2[item], color=c2)
        
ax.legend([bp1["boxes"][0],bp2["boxes"][0]], ['Survived', 'Died'], loc='upper right')
plt.ylabel("Ages")
plt.title("Analysis in age and smoking on survival status")
plt.savefig("age_smoking_survival status.png")
plt.show()
plt.close()
"""Analysis in age and diabetes on survival status"""
fig, ax=plt.subplots()
live_ageD = live_df.loc[live_df["diabetes"]== 1, "age"].values
live_ageND = live_df.loc[live_df["diabetes"]==0, "age"].values
dead_ageD = death_df.loc[death_df["diabetes"]==1, "age"].values
dead_ageND =death_df.loc[death_df["diabetes"]==0, "age"].values
live_ages = [live_ageD, live_ageND]
dead_ages = [dead_ageD, dead_ageND]
bp1 = ax.boxplot(live_ages, positions=[1,4], widths=0.35, patch_artist = True)
bp2 = ax.boxplot(dead_ages, positions=[2,5], widths=0.35, patch_artist = True)
ax.set_xticklabels(['Diabetic', 'Not diabetic'])
ax.set_xticks([1.5,4.5])
c1="red"
c2="blue"
for item in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bp1[item], color=c1)
        plt.setp(bp2[item], color=c2)
ax.legend([bp1["boxes"][0],bp2["boxes"][0]], ['Survived', 'Died'], loc='upper right')
plt.ylabel("Ages")
plt.title("Analysis in age and diabetes on survival status")
plt.savefig("age_diabetes_survival status.png")
plt.show()
plt.close()

labels = 'Diabetic Deaths', 'Non diabetic deaths', 'Diabetic survivors','non diabetic survivors'
dd = death_df.loc[death_df["diabetes"]==1].shape[0]
ndd = death_df.loc[death_df["diabetes"]==0].shape[0]
ds = live_df.loc[live_df["diabetes"]==1].shape[0]
nds = live_df.loc[live_df["diabetes"]==0].shape[0]
sizes = [dd, ndd, ds, nds]
explode=(0.1, 0.1, 0.1, 0.1)
plt.pie(sizes, explode =explode,labels=labels, autopct='%1.1f%%')
plt.axis("equal")
plt.savefig("Diabetic_survival_pie_chart.png")
plt.show()
plt.close()

"""creatinine phosphokinase and survival rate"""
creaS = live_df["creatinine_phosphokinase"].values
creaD = death_df["creatinine_phosphokinase"].values
plt.hist(creaS, label ="Survived", alpha=1)
plt.hist(creaD, label="Died", alpha=0.8)
plt.legend()
plt.ylabel("count")
plt.xlabel("Number of creatinine_phosphokinase")
plt.savefig("creatinine_phosphokinase_survival_rate.png")
plt.show()
plt.close()
"""ejection fraction and survival rate"""
ejecS = live_df["ejection_fraction"].values
ejecD = death_df["ejection_fraction"].values
plt.hist(ejecS, label ="Survived", alpha=1)
plt.hist(ejecD, label="Died", alpha=0.8)
plt.legend()
plt.ylabel("count")
plt.xlabel("Number of ejection_fraction")
plt.savefig("ejection_fraction_survival_rate.png")
plt.show()
plt.close()
"""platelets and survival rate"""
platS = live_df["platelets"].values
platD = death_df["platelets"].values
sns.distplot(platS, label ="Survived", norm_hist=True, kde=True) #removing the normlised form is not working
sns.distplot(platD, label="Died", norm_hist=True, kde=True) #removing the normlised form is not working
plt.legend()
plt.ylabel("count")
plt.xlabel("Number of platelets")
plt.savefig("platelets_survival_rate.png")
plt.show()
plt.close()
"""Serum Creatinine and survival rate"""
serumS = live_df["serum_creatinine"].values
serumD = death_df["serum_creatinine"].values
sns.distplot(serumS, label ="Survived", norm_hist=True, kde=True) #removing the normlised form is not working
sns.distplot(serumD, label="Died", norm_hist=True, kde=True) #removing the normlised form is not working
plt.legend()
plt.ylabel("count")
plt.xlabel("Number of Serum creatinine")
plt.savefig("Serum_creatinine_survival_rate.png")
plt.show()
plt.close()




