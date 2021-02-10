#
import numpy as np

cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2471, 0.2435, 0.2616])
upper_limit = ((1 - cifar10_mean) / cifar10_std)
lower_limit = ((0 - cifar10_mean) / cifar10_std)