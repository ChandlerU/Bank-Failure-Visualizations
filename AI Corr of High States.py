import pandas as pd
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("bank-data_2000-2019.csv")
ef = pd.read_csv("all-institutions.csv", usecols = ['CHANGEC1', 'CHANGEC2', 'CHANGEC3', 'CHANGEC4', 'CHANGEC5', 'CERT', 'STALP', 'ACTIVE', 'BKCLASS', 'REGAGNT', 'INSAGNT1', 'CHRTAGNT'])
e = 0


for i in df.iterrows():
    if str(df.at[e,'CITYST'])[-2:] == 'FL' or str(df.at[e,'CITYST'])[-2:] == 'GA' or str(df.at[e,'CITYST'])[-2:] == 'IL' or str(df.at[e,'CITYST'])[-2:] == 'CA':
        pass
    elif df.at[e,'RESTYPE'] == 'ASSISTANCE':
        df.drop(e, inplace = True)
    else:
        df.drop(e, inplace = True)
    e += 1
cert_l = list(df.CERT)
e = 0

#Drop all banks that are not in states of interest or are closed but didn't necessarily fail. 
for i in ef.iterrows():
    if (ef.at[e, 'CERT'] not in cert_l) and (ef.at[e, 'ACTIVE'] == 0):
        ef.drop(e, inplace = True)
    elif str(ef.at[e,'STALP']) == 'FL' or str(ef.at[e,'STALP']) == 'GA' or str(ef.at[e,'STALP']) == 'IL' or str(ef.at[e,'STALP']) == 'CA':
        pass
    else:
        ef.drop(e, inplace = True)
    e += 1

#Split the data frame in two. One frame containing Failures and the other Survivors
efs1 = ef[ef['ACTIVE'] == 1]
efs2 = ef[ef['ACTIVE'] == 0]
print(ef)
plt.rcParams['figure.figsize'] = (10, 5)

# make subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# make the data read to feed into the visulizer
X_Pclass = efs1.groupby('BKCLASS').size().reset_index(name='Counts')['BKCLASS']
Y_Pclass = efs1.groupby('BKCLASS').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_Pclass, Y_Pclass)
axes[0, 1].set_title('BKCLASS of Active', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)


# make the data read to feed into the visulizer
X_Pclass = efs2.groupby('BKCLASS').size().reset_index(name='Counts')['BKCLASS']
Y_Pclass = efs2.groupby('BKCLASS').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_Pclass, Y_Pclass)
axes[0, 0].set_title('BKCLASS of Failed', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

#
X_Pclass = efs1.groupby('CHRTAGNT').size().reset_index(name='Counts')['CHRTAGNT']
Y_Pclass = efs1.groupby('CHRTAGNT').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_Pclass, Y_Pclass)
axes[1, 1].set_title('CHRTAGNT of Active', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)


# make the data read to feed into the visulizer
X_Pclass = efs2.groupby('CHRTAGNT').size().reset_index(name='Counts')['CHRTAGNT']
Y_Pclass = efs2.groupby('CHRTAGNT').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_Pclass, Y_Pclass)
axes[1, 0].set_title('CHRTAGNT of Failed', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)
### make the data read to feed into the visulizer
##X_Sex = data.groupby('Sex').size().reset_index(name='Counts')['Sex']
##Y_Sex = data.groupby('Sex').size().reset_index(name='Counts')['Counts']
### make the bar plot
##axes[1, 0].bar(X_Sex, Y_Sex)
##axes[1, 0].set_title('Sex', fontsize=25)
##axes[1, 0].set_ylabel('Counts', fontsize=20)
##axes[1, 0].tick_params(axis='both', labelsize=15)

plt.show()
### make the data read to feed into the visulizer
##X_Embarked = data.groupby('Embarked').size().reset_index(name='Counts')['Embarked']
##Y_Embarked = data.groupby('Embarked').size().reset_index(name='Counts')['Counts']
### make the bar plot
##axes[1, 1].bar(X_Embarked, Y_Embarked)
##axes[1, 1].set_title('Embarked', fontsize=25)
##axes[1, 1].set_ylabel('Counts', fontsize=20)
##axes[1, 1].tick_params(axis='both', labelsize=15)
##
##le = preprocessing.LabelEncoder()
##ef['STALP'] = le.fit_transform(ef['STALP'])
##ef['BKCLASS'] = le.fit_transform(ef['BKCLASS'])
##ef['REGAGNT'] = le.fit_transform(ef['REGAGNT'])
##ef['INSAGNT1'] = le.fit_transform(ef['INSAGNT1'])
##ef['CHRTAGNT'] = le.fit_transform(ef['CHRTAGNT'])
##features_model = ['STALP', 'BKCLASS', 'REGAGNT', 'INSAGNT1', 'CHRTAGNT','CHANGEC1', 'CHANGEC2', 'CHANGEC3', 'CHANGEC4', 'CHANGEC5']
##data_model_X = pd.concat([ef[features_model], ef], axis=1)
##data_model_y = ef.replace({'INACTIVE': {0: 'Success', 1: 'Failure'}})['INACTIVE']
##X_train, X_val, y_train, y_val = train_test_split(data_model_X, data_model_y, test_size =0.3, random_state=11)
##print("No. of samples in training set: ", X_train.shape[0])
##print("No. of samples in validation set:", X_val.shape[0])
##
### Survived and not-survived
##print('\n')
##print('No. of survived and not-survived in the training set:')
##print(y_train.value_counts())
##
##print('\n')
##print('No. of survived and not-survived in the validation set:')
##print(y_val.value_counts())
##
##from sklearn.linear_model import LogisticRegression
##
##from yellowbrick.classifier import ConfusionMatrix
##from yellowbrick.classifier import ClassificationReport
##from yellowbrick.classifier import ROCAUC
##
### Instantiate the classification model 
##model = LogisticRegression()
##
###The ConfusionMatrix visualizer taxes a model
##classes = ['Failure','Success']
##cm = ConfusionMatrix(model, classes=classes, percent=False)
##
###Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
##cm.fit(X_train, y_train)
##
###To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
###and then creates the confusion_matrix from scikit learn.
##cm.score(X_val, y_val)
##
### change fontsize of the labels in the figure
##for label in cm.ax.texts:
##    label.set_size(20)
##
###How did we do?
##cm.poof()
##plt.rcParams['figure.figsize'] = (15, 7)
##plt.rcParams['font.size'] = 20
