import numpy as np
import pandas as pd
import plotly.express as px

def weighted_radius(Md,Rd,Mb,Rb):
    Mtot = Md+Mb
    return Md/Mtot*0.68*Rd + Mb/Mtot*Rb

df = pd.read_hdf('tng300-sam.h5')
df['GalpropWeightedRadius'] = weighted_radius(df['GalpropMdisk'],
                df['GalpropRdisk'],df['GalpropMbulge'],df['GalpropRbulge'])
dfs = df.sample(n=5000)

fig = px.scatter(dfs, x='GalpropHalfmassRadius', y = 'GalpropWeightedRadius',
                 color='GalpropMstar')
fig.update_layout(yaxis_range=[0,40],xaxis_range=[0,40])
fig.show()