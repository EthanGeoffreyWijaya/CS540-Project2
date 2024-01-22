from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    data = np.load(filename)
    data = data - np.mean(data, axis=0)
    return data

def get_covariance(dataset):
    #matrix = np.dot(dataset, np.transpose(dataset))
    matrix = np.dot(np.transpose(dataset), dataset)
    return np.transpose(matrix) / (len(dataset) - 1)

def get_eig(S, m):
    n = len(S)
    lm, u = eigh(S, subset_by_index=[n-m, n-1])
    lm = lm[::-1]
    evals = np.zeros((m, m))
    for i in range(m):
        evals[i][i] = lm[i]
    return evals, np.fliplr(u)

def get_eig_prop(S, prop):
    sumval = np.sum(eigh(S, eigvals_only=True))
    lm, u = eigh(S, subset_by_value=[prop * sumval, np.inf])
    lm = lm[::-1]
    m = len(lm)
    evals = np.zeros((m, m))
    for i in range(m):
        evals[i][i] = lm[i]
    return evals, np.fliplr(u)

def project_image(image, U):
    x = np.copy(U)
    for i in range(len(U[0])):
        a = np.dot(np.transpose(x[:, i]), image)
        x[:, i] = np.dot(a, x[:, i])
    return x.sum(axis=1)

def display_image(orig, proj):
    oldim = np.transpose(np.reshape(orig, (32, 32)))
    newim = np.transpose(np.reshape(proj, (32, 32)))
    fig, (im1, im2) = plt.subplots(1, 2)
    im1.set_title("Original")
    im2.set_title("Projection")
    corig = im1.imshow(oldim, aspect="equal")
    cproj = im2.imshow(newim, aspect="equal")
    fig.colorbar(corig, ax=im1)
    fig.colorbar(cproj, ax=im2)
    plt.show()
