
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as st
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

import sys
np.set_printoptions(threshold=sys.maxsize)

filepath = r'D:\magisterka\normokapnia_projekt.csv'
file = pd.read_csv(filepath)

"AMP_spec,AMP_ptp,PI_spec,PI_ptp"
AMP_spec = file['AMP_spec'].to_numpy()
AMP_ptp = file['AMP_ptp'].to_numpy()
PI_spec = file['PI_spec'].to_numpy()
PI_ptp = file['PI_ptp'].to_numpy()
PI = np.empty([len(PI_spec), 2])
AMP = np.empty([len(AMP_spec), 2])

for i in range(0, len(PI_spec)):
    PI[i] = [PI_ptp[i], PI_spec[i]]
    AMP[i] = [AMP_ptp[i], AMP_spec[i]]

param = AMP
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto", algorithm="lloyd").fit(param)
label = kmeans.fit_predict(param)
centroid = kmeans.cluster_centers_

filtered_label0 = param[label == 0]
filtered_label1 = param[label == 1]
filtered_label2 = param[label == 2]
filtered_label3 = param[label == 3]
filtered_label4 = param[label == 4]
plt.figure()
plt.scatter(centroid[:, 0], centroid[:, 1], marker='*')
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], marker='^')
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], marker='^')
plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1])
plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1])
plt.scatter(filtered_label4[:, 0], filtered_label4[:, 1])
plt.title("n = 3")
plt.xlabel("AMP_ptp")
plt.ylabel("AMP_spect")

spectral = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=0).fit(param)
label = spectral.fit_predict(param)

filtered_label0 = param[label == 0]
filtered_label1 = param[label == 1]
filtered_label2 = param[label == 2]
filtered_label3 = param[label == 3]
filtered_label4 = param[label == 4]
plt.figure()
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1])
plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1])
plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1])
plt.scatter(filtered_label4[:, 0], filtered_label4[:, 1])
plt.title("n = 3")
plt.xlabel("PI_ptp")
plt.ylabel("PI_spect")

gm = GaussianMixture(n_components=3, random_state=0).fit(param)
label = gm.fit_predict(param)
filtered_label0 = param[label == 0]
filtered_label1 = param[label == 1]
filtered_label2 = param[label == 2]
filtered_label3 = param[label == 3]
filtered_label4 = param[label == 4]
plt.figure()
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1])
plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1])
plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1])
plt.scatter(filtered_label4[:, 0], filtered_label4[:, 1])
plt.title("n = 3")
plt.xlabel("AMP_ptp")
plt.ylabel("AMP_spect")

# korelacje amplitud
x_AMP = st.add_constant(AMP_ptp)
model = st.OLS(AMP_spec, x_AMP)
results = model.fit()
print("podsumowanie", results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)

# korelacja klastrow
y_AMP = st.add_constant(AMP_ptp[label == 0])
model = st.OLS(AMP_spec[label == 0], y_AMP)
results = model.fit()
print("podsumowanie", results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)

# korelacje PI
x_PI = st.add_constant(PI_ptp)
model = st.OLS(PI_spec, x_PI)
results = model.fit()
print("podsumowanie", results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)

# plotowanie regresji
plt.figure()
sb.regplot(x=AMP_ptp, y=AMP_spec)
plt.xlabel("AMP_CBFV peak to peak")
plt.ylabel("AMP_CBFV spectral")
#
plt.figure()
sb.regplot(x=PI_ptp, y=PI_spec)
plt.xlabel("PI peak to peak")
plt.ylabel("PI spectral")
#
# plt.figure()
# sb.regplot(x=AMP_ptp[label == 0], y=AMP_spec[label == 0])
# sb.regplot(x=AMP_ptp[label == 1], y=AMP_spec[label == 1])

#
# bland altmann
# diff_AMP_spectral_arr = np.asarray(AMP_spec)
# diff_AMP_ptp_arr = np.asarray(AMP_ptp)
# k, ax = plt.subplots(1, figsize=(8, 5))
# st.graphics.mean_diff_plot(diff_AMP_ptp_arr, diff_AMP_spectral_arr, ax=ax)
# ax.set_title("AMP_CBFV")
# pi_specrtral_arr = np.asarray(PI_spec)
# pi_ptp_arr = np.asarray(PI_ptp)
# g, ax = plt.subplots(1, figsize=(8, 5))
# st.graphics.mean_diff_plot(pi_specrtral_arr, pi_ptp_arr, ax=ax)
# ax.set_title("PI")

plt.show()
