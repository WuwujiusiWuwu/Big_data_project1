import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
df= pd.read_csv("new_data.csv")
sns.set(font_scale=1)
plt.rc('font',family='Times New Roman',size=5)

sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.savefig('fix.jpg', dpi=800)






