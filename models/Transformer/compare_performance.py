import matplotlib.pyplot as plt
import pickle

import pandas as pd

loss_transformer_z = pd.read_csv('log_mac_all.csv')
loss_transformer = pickle.load(open("losses_11092318.pkl", 'rb'))
loss_deepar = pickle.load(open("../DeepAR/losses_11101126.pkl", 'rb'))
loss_deepfactor = pickle.load(open("../DeepFactors/losses_11101339.pkl", 'rb'))
loss_MQRNN = pickle.load(open("../MQRNN/losses_11101128.pkl", 'rb'))

plt.plot(loss_transformer_z['train loss(mse)'], label='transformer_z')
plt.plot([loss_transformer[i] for i in range(0, len(loss_transformer), int(len(loss_transformer)/50))], label='transformer')
plt.plot([loss_deepar[i] for i in range(0, len(loss_deepar), int(len(loss_deepar)/50))], label='deepar')
plt.plot([loss_deepfactor[i] for i in range(0, len(loss_deepfactor), int(len(loss_deepfactor)/50))], label='deepfactors')
plt.plot([loss_MQRNN[i] for i in range(0, len(loss_MQRNN), int(len(loss_MQRNN)/50))], label='mqrnn')
plt.xlabel("Period")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(loss_transformer_z['train loss(mse)'], label='transformer_z')
plt.plot([loss_transformer[i] for i in range(0, len(loss_transformer), int(len(loss_transformer)/50))], label='transformer')
plt.plot([loss_MQRNN[i] for i in range(0, len(loss_MQRNN), int(len(loss_MQRNN)/50))], label='mqrnn')
plt.xlabel("Period")
plt.ylabel("Loss")
plt.legend()
plt.show()