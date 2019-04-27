pip install seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train.csv')
train

test= pd.read_csv('test.csv')
test

print(train.info())
print(test.info())

passengerid= test.filter(['PassengerId'])
passengerid

total_nulls= train.isnull().sum().sort_values(ascending = False)
null_perc= round(( train.isnull().sum().sort_values(ascending = False))*100/len(train), 2)
nulls_check_train=pd.concat([total_nulls,null_perc], axis=1, keys= ['Total','Percent'])
nulls_check_train

### similarly calculating missing values in test set
total_nulls= test.isnull().sum().sort_values(ascending = False)
nulls_perc= round((test.isnull().sum().sort_values(ascending = False))*100/len(test),2)
nulls_check_test= pd.concat([total_nulls,nulls_perc], axis =1 , keys=['total','Percentage'])
nulls_check_test


train.groupby(['Embarked']).count()


total_vals= pd.DataFrame(train.Embarked.value_counts(dropna=False))
perc_vals= pd.DataFrame(round((train.Embarked.value_counts(dropna= False))*100/len(train),2))
embarked_nulls= pd.concat([total_vals,perc_vals], axis=1,keys=['total','percent'])
embarked_nulls

train[train.Embarked.isnull()]

fig, ax= plt.subplots(figsize=(16,12), ncols=2)
ax1= sns.boxplot(x= 'Embarked', y= 'Fare', hue= 'Pclass', data=train, ax= ax[0]);
ax2= sns.boxplot(x= 'Embarked', y= 'Fare', hue= 'Pclass', data=test, ax= ax[1]);
ax1.set_title('Training Data', fontsize= 18)
ax2.set_title('Test Data', fontsize= 18)
fig.show()

train.Embarked.fillna("C", inplace=True)
values=['62','830']
train.loc[train['PassengerId'].isin(values)]

train_survived= train.Survived
train.drop(['Survived'], axis =1, inplace=True)

combined=pd.concat([train,test], ignore_index=False)
combined.Cabin.fillna('N', inplace=True)    
combined

combined.Cabin=[i[0] for i in combined.Cabin]
withN= combined[combined.Cabin=='N']
withoutN= combined[combined.Cabin!='N']
combined.groupby(['Cabin'])['Fare'].mean().sort_values()


def cabin_estimator(i):
    a=0
    if i<=(16):
        a='G'
    elif i>=17 and i<=26:
        a='F'
    elif i>26 and i<=38:
        a='T'
    elif i>38 and i<=49:
        a='A'
    elif i>49 and i<=53:
        a='D'
    elif i>53 and i<=107:
        a='E'
    elif i>107 and i<=122:
        a='A'    
    else:
        a='B'
    return a


 withN['Cabin']= withN.Fare.apply(lambda x:cabin_estimator(x))
withN

combined= pd.concat([withN, withoutN], axis=0)
combined.sort_values('PassengerId', ascending=True, inplace=True)
train=combined[:891]
test=combined[892:]
print(str(train.count())+" "+str(test.count()))


test[test.Fare.isnull()]

missing_value= test[(test.Pclass==3) & (test.Sex=='Male') & (test.Embarked=='S')].Fare.mean()
missing_value


pal = {'male':"green", 'female':"Pink"}
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived", 
            data=train, 
            palette = pal,
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);


plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train, 
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);


train[['Pclass', 'Survived']].groupby("Pclass").mean().reset_index()


survived_summary = train.groupby("Survived")
survived_summary.mean().reset_index()

survived_summary = train.groupby("Sex")
survived_summary.mean().reset_index()

survived_summary = train.groupby("Pclass")
survived_summary.mean().reset_index()


#I have gathered a small summary from the statistical overview above. Let's see what they are...

#This data set has 891 raw and 9 columns.
#only 38% passenger survived during that tragedy.
#~74% female passenger survived, while only ~19% male passenger survived.
#~63% first class passengers survived, while only 24% lower class passenger survived.



pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))


corr = train.corr()**2
corr.Survived.sort_values(ascending=False)

male_mean = train[train['Sex'] == 1].Survived.mean()

female_mean = train[train['Sex'] == 0].Survived.mean()
print ("Male survival mean: " + str(male_mean))
print ("female survival mean: " + str(female_mean))

print ("The mean difference between male and female survival rate: " + str(female_mean - male_mean))


male = train[train['Sex'] == 1]
female = train[train['Sex'] == 0]

# getting 50 random sample for male and female. 
import random
male_sample = random.sample(list(male['Survived']),50)
female_sample = random.sample(list(female['Survived']),50)

# Taking a sample means of survival feature from male and female
male_sample_mean = np.mean(male_sample)
female_sample_mean = np.mean(female_sample)

# Print them out
print ("Male sample mean: " + str(male_sample_mean))
print ("Female sample mean: " + str(female_sample_mean))
print ("Difference between male and female sample mean: " + str(female_sample_mean - male_sample_mean))


import scipy.stats as stats

print (stats.ttest_ind(male_sample, female_sample))
print ("This is the p-value when we break it into standard form: " + format(stats.ttest_ind(male_sample, female_sample).pvalue, '.32f'))

# Creating a new colomn with a 
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]

def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)


## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)


# separating our independent and dependent variable
X = train.drop(['Survived'], axis = 1)
y = train["Survived"]


#age_filled_data_nor = NuclearNormMinimization().complete(df1)
#Data_1 = pd.DataFrame(age_filled_data, columns = df1.columns)
#pd.DataFrame(zip(Data["Age"],Data_1["Age"],df["Age"]))


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = .33, random_state = 0)


train.sample(5)


# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

## transforming "train_x"
train_x = sc.fit_transform(train_x)
## transforming "test_x"
test_x = sc.transform(test_x)

## transforming "The testset"
test = sc.transform(test)
pd.DataFrame(train_x, columns=headers).head()


# import LogisticRegression model in python. 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score


## call on the model object
logreg = LogisticRegression(solver='lbfgs')

## fit the model with "train_x" and "train_y"
logreg.fit(train_x,train_y)

## Once the model is trained we want to find out how well the model is performing, so we test the model. 
## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome. 
y_pred = logreg.predict(test_x)

## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing. 

print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_pred, test_y),4)))


from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(test_x)

FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
plt.show()

