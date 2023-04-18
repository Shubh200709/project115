import numpy as np
import pandas as pd
import matplotlib.pyplot as plp
from sklearn.linear_model import LogisticRegression
import plotly.express as px

data = pd.read_csv('escape_velocity.csv')
vel = data['Velocity'].to_list()
esc = data['Escaped'].to_list()

vel_array = np.array(vel)
esc_array = np.array(esc)

m,c = np.polyfit(vel,esc,1)

y_array = []
for i in vel_array:
    y = m*i + c
    y_array.append(y)

scatter = px.scatter(x=vel_array,y=esc_array)
scatter.update_layout(shapes=[dict(type='line',y0=min(esc_array),y1=max(esc_array),x0=min(vel_array),x1=max(vel_array))])
scatter.show()

def model(x):
    return 1/(1+np.exp(-x))

x = np.reshape(vel_array,(len(vel_array),1))
y = np.reshape(esc_array,(len(esc_array),1))

logreg = LogisticRegression()
logreg.fit(x,y)

x_test = np.linspace(0,5000,10000)
chance = model(x_test*logreg.coef_+logreg.intercept_).ravel()

plp.plot(x_test,chance,color='red',linewidth=3)
plp.axhline(y=0,color='k',linestyle='-')
plp.axhline(y=1,color='k',linestyle='-')
plp.axhline(y=0.5,color='b',linestyle='--')

plp.axvline(x=x_test[46],color='b',linestyle='--')

plp.xlabel('x-axis')
plp.ylabel('y-axis')
plp.xlim(0,30)
plp.show()
