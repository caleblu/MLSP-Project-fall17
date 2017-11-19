from sklearn.mixture import GaussianMixture
import numpy as np

def train_gmm(jnt):
    """
    train gmm with joint vector
    :param jnt: joint vector of spectral features
    :return: trained gmm model
    """
    gmm = GaussianMixture(n_components=32, covariance_type='full', max_iter=100)
    gmm.fit(jnt)

def get_density_x(src, gmm):
    pass

def get_mean_tgt(gmm):
    pass

def get_cross_cov(gmm):
    pass

def predict(src, gmm):
    """
    predict target value given src spectral features
    :param src:
    :param gmm:
    :return:
    """
    m = gmm.n_components
    y = np.zeros(src.shape)
    density_x = get_density_x(src, gmm)
    v = get_mean_tgt(gmm)
    diag = get_cross_cov(gmm)
    for i in range(m):
        y = y + density_x[i] * v[i] + diag[i] *
