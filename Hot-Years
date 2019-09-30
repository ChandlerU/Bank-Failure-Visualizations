import pandas as pd
import matplotlib.pyplot as plt

#Read in the data
df = pd.read_csv("bank-data_2000-2019.csv")

#Only want failure year for line graph
e = 0
for i in df.iterrows():
    df.at[e,'FAILDATE'] = str(df.at[e,'FAILDATE'])[-4:]
    e += 1
df['RESTYPE'] = df.RESTYPE.replace('FAILURE',1)
df['RESTYPE'] = df.RESTYPE.replace('ASSISTANCE',1)
df.RESTYPE = df.RESTYPE.astype(int)

#Calculate total banks by year
gf = df.groupby('FAILDATE').sum()
h = list(df.FAILDATE.unique())

#No banks failed in 05 or 06. Should reflect in visual 
h.insert(4, '2005')
h.insert(5, '2006')
h.sort()
b = list(gf['RESTYPE'])
b.insert(5, 0)
b.insert(6, 0)

#Make the line graph
plt.rcParams['figure.figsize'] = (10, 5)
plt.plot(h, b)
plt.ylabel('Amount of Bank Failures')
plt.xlabel('Years')
plt.show()
