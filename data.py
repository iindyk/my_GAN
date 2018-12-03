import numpy as np


def get_toy_data(n, m):
    random_flips = 0.1
    dataset = np.random.uniform(0, 1, (n, m))
    labels = []
    for i in range(n):
        if sum(dataset[i, :]) > 0.5*m:
            labels.append(1)
        else:
            labels.append(-1)
    # random attack
    indices = np.random.randint(n, size=int(random_flips*n))
    for i in indices:
        if labels[i] == 1:
            labels[i] = -1
        else:
            labels[i] = 1
    return dataset[:int(0.5*n), :], labels[:int(0.5*n)], dataset[int(0.5*n):, :], labels[int(0.5*n):]