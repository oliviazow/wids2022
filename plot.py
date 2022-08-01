from matplotlib import pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

x = [10.32,9.58,8.01,9.14,10.74,8.59,8.99,8.10,11.26]
y = [0.27,0.07,0.00,0.01,0.14,0.03,0.01,0.02,0.11]
sc = plt.scatter(x, y, alpha=0.8, s=100,color="red")
for i, label in enumerate(['0043', '0071', '0073', '0075', '0298', '0540', '0561', '1019', '1501']):
    plt.annotate(label, (x[i], y[i]), fontsize=14, rotation=30)
plt.xlabel('Total Correlation')
plt.ylabel('R^2 Change')
# plt.legend(*sc.legend_elements("sizes", num=6))
plt.show()