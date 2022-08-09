import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
columns = ['5%', '10%', '15%', '20%']
index = ['UMEAN', 'IMEAN', 'UPCC', 'IPCC', 'UIPCC', 'NMF', 'PMF', 'NCF', 'LDCF', 'FedNCF', 'FedLDCF']
data_rt_mae = [
    [0.876, 0.875, 0.875, 0.875],
    [0.702, 0.688, 0.683, 0.680],
    [0.652, 0.548, 0.509, 0.475],
    [0.692, 0.666, 0.542, 0.486],
    [0.625, 0.582, 0.501, 0.450],
    [0.546, 0.478, 0.447, 0.427],
    [0.569, 0.486, 0.452, 0.430],

    [0.443, 0.389, 0.373, 0.361],
    [0.410, 0.369, 0.346, 0.330],

    [0.549, 0.480, 0.453, 0.435],
    [0.501, 0.448, 0.428, 0.398],

]
data_rt_rmse = [
    [1.853, 1.856, 1.856, 1.855],
    [1.567, 1.543, 1.533, 1.528],
    [1.672, 1.509, 1.457, 1.404],
    [1.400, 1.374, 1.257, 1.211],
    [1.388, 1.330, 1.250, 1.200],
    [1.473, 1.283, 1.202, 1.160],
    [1.537, 1.316, 1.220, 1.169],

    [1.331, 1.302, 1.254, 1.203],
    [1.281, 1.238, 1.172, 1.140],

    [1.621, 1.594, 1.560, 1.467],
    [1.546, 1.529, 1.438, 1.373],

]
data_tp_mae = [
    [53.9303, 53.8403, 53.6939, 53.9565],
    [26.8934, 26.8934, 26.6892, 26.6262],
    [31.4333, 24.7058, 22.3486, 21.2129],
    [29.4089, 28.9753, 27.2390, 26.8617],
    [29.7889, 22.8508, 19.5598, 17.8447],
    [18.8847, 15.5785, 14.3463, 13.5870],
    [19.0788, 15.9950, 14.6821, 13.9223],
    [21.5517, 15.8463, 13.8349, 13.0263], ]
data_tp_rmse = [
    [110.5447, 110.4771, 110.6045, 110.0648],
    [66.1061, 64.7624, 64.4731, 64.0785],
    [77.0888, 64.1858, 58.9557, 56.1601],
    [65.8795, 62.3999, 58.1607, 56.6058],
    [71.6566, 61.4218, 54.6422, 50.3597],
    [57.5301, 47.8238, 44.0323, 41.7636],
    [57.8884, 48.0795, 44.0503, 41.7244],
    [62.0946, 52.8056, 48.0702, 45.9650], ]

data_fed_mae = [
    [0.443, 0.389, 0.373, 0.361],
    [0.410, 0.369, 0.346, 0.330],

    [0.549, 0.480, 0.453, 0.435],
    [0.501, 0.448, 0.428, 0.398],
]
data_fed_rmse = [
    [1.331, 1.302, 1.254, 1.203],
    [1.281, 1.238, 1.172, 1.140],

    [1.621, 1.594, 1.560, 1.467],
    [1.546, 1.529, 1.438, 1.373],
]
index_fed = ['NCF', 'LDCF', 'FedNCF', 'FedLDCF']

data_fraction = [
    # [0.573, 0.561, 0.553, 0.549],
    # [0.525, 0.512, 0.505, 0.501],
    #
    [1.630, 1.627, 1.624, 1.621],
    [1.557, 1.553, 1.549, 1.546],
]
index_fraction = ['FedNCF', 'FedLDCF']
columns_fraction = ['0.25', '0.50', '0.75', '1.00']


df_rt_mae = pd.DataFrame(data_rt_mae, columns=columns, index=index)
df_rt_rmse = pd.DataFrame(data_rt_rmse, columns=columns, index=index)

df_fed_mae = pd.DataFrame(data_fed_mae, columns=columns, index=index_fed)
df_fed_rmse = pd.DataFrame(data_fed_rmse, columns=columns, index=index_fed)

df_frac = pd.DataFrame(data_fraction, columns=columns_fraction, index=index_fraction)
# fig, x = plt.subplots()
ax = sns.lineplot(data=df_frac.T, markers=True)
ax.set_xlabel('Fraction')
ax.set_ylabel('RMSE')
ax.set_title('Response Time')

plt.legend(loc='upper right')
# plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
# fig.subplots_adjust(right=0.75)

plt.show()
