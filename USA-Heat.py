import pandas as pd
import plotly.graph_objects as go

#Read in the data
df = pd.read_csv("bank-data_2000-2019.csv")

#Only want to look at bank failures by state for now.
e = 0
for i in df.iterrows():
    df.at[e,'CITYST'] = str(df.at[e,'CITYST'])[-2:]
    if df.at[e,'RESTYPE'] == 'ASSISTANCE':
        df.drop(e, inplace = True)
    e+=1

#To calculate the amount of failures we need a metric
df['RESTYPE'] = df.RESTYPE.replace('FAILURE',1)
df.RESTYPE = df.RESTYPE.astype(int)

#Aggregate rows by CITYST to get count of closures by state
gf = df.groupby('CITYST').sum()
do = df.CITYST.unique()
do.sort()

#Make the map
fig = go.Figure(data=go.Choropleth(
    locations=do, 
    z = gf['RESTYPE'].astype(int), 
    locationmode = 'USA-states', 
    colorscale = 'greens',
    
))

fig.update_layout(
    title_text = 'Number of Bank Failures by State Since 2000 (Almost 50% occur in CA, FL, GA, or IL)',
    geo_scope='usa',
)

fig.show()
