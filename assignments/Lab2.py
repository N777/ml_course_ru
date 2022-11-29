import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.DataFrame({"x1": np.linspace(2, 500, 500),
                   "x2": np.linspace(2, 30, 500)})

# y = np.cos(x1) * np.power(x2, 3)
df.insert(loc=len(df.columns), column='y', value=5 * np.log(df['x1']) * np.power(df['x2'], 2))
df.head(30)
df.to_csv("out.csv", index=False)
df.plot(x='x1', y='y', )
plt.plot(df['x1'], df['y'], 'ro')
df.plot(x='x2', y='y')
df.describe()
parsed_df = df[df['x1'] < df['x1'].mean()]
parsed_df.to_csv("out_parsed.csv", index=False)

x1 = np.array(df['x1']).flatten()
x2 = np.array(df['x2']).flatten()

x1, x2 = np.meshgrid(x1, x2)

z = 5 * np.log(x1) * np.power(x2, 2)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(x1, x2, z, linewidth=0.1, antialiased=True)


plt.show()
