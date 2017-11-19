from sklearn.mixture import GaussianMixture
import numpy as np

def train_GMM(jnt):
    """
    train GMM with joint vectors
    :param jnt: joint vector of spectral features
    :return: trained gmm model
    """
    gmm = GaussianMixture(n_components=10, covariance_type='full')
    gmm.fit(jnt)
    return gmm

def get_density_x(src, gmm):
    """
    :param src: source spectral features
    :param gmm: trained GMM model
    :return: get the conditional probability that src belong to a component
    """
    mid = gmm.means_.shape[1] / 2
    x_mean = gmm.means_[:, 0:mid]
    x_cov = gmm.covariances_[:, :mid, :mid]
    x_gmm = GaussianMixture(n_components=10, covariance_type='full')
    x_gmm.covariances_ = x_cov
    x_gmm.means_ = x_mean
    return x_gmm.predict_proba(src)

def get_mean_tgt(gmm):
    """
    :param gmm: trained GMM model
    :return: get the mean of the target spectral features of each component
    """
    mid = gmm.means_.shape[1] / 2
    y_mean = gmm.means_[:, mid:]
    return y_mean

def get_cross_cov(gmm):
    # TODO use full conversion instead of VQ
    pass

def predict(src, gmm):
    """
    predict target value given src spectral features
    :param src: source spectral features
    :param gmm: trained GMM model
    :return: predicted target spectral features
    """
    m = gmm.n_components
    y = np.zeros(src.shape)
    density_x = get_density_x(src, gmm)
    v = get_mean_tgt(gmm)
    for i in range(m):
        y = y + density_x[i] * v[i]
    return y

if __name__=="__main__":
    pass