import numpy as np
import matplotlib.pyplot as plt

# psnr = np.load('psnr.npy')
# mask_psnr = np.load('mask_psnr.npy')
# x = np.arange(len(psnr))
# plt.figure(figsize=(12, 3))
# plt.subplot(131)
# plt.plot(x, psnr)
# plt.subplot(132)
# plt.plot(x, mask_psnr)
# plt.subplot(133)
# plt.plot(x, mask_psnr - psnr)
# plt.savefig('test2.png')
# # print(np.max(psnr))
# print(psnr.sum()/100, mask_psnr.sum()/100)
beta = []
for i in range(200):
    beta.append(1 / (1 + 0.02 * i))
beta = np.array(beta)
x = np.arange(len(beta))
plt.plot(x, beta)
plt.savefig('beta')
