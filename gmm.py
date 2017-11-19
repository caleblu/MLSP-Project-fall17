import sklearn.mixture
import numpy as np

def train_GMM(jnt):
    """
    train GMM with joint vectors
    :param jnt: joint vector of spectral features
    :return: trained GMM model
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=10, covariance_type='full')
    GMM.fit(jnt)
    return GMM

def get_density_x(src, GMM):
    """
    :param src: source spectral features
    :param GMM: trained GMM model
    :return: get the conditional probability that src belong to a component
    """
    mid = GMM.means_.shape[1] / 2
    x_GMM = sklearn.mixture.GaussianMixture(n_components=10, covariance_type='full')
    x_GMM.covariances_ = GMM.covariances_[:, :mid, :mid]
    x_GMM.means_ = GMM.means_[:, 0:mid]
    x_GMM.weights_ = GMM.weights_
    x_GMM.precisions_cholesky_ =  sklearn.mixture.gaussian_mixture._compute_precision_cholesky(x_GMM.covariances_, 'full')
    return x_GMM.predict_proba(src)

def get_mean_tgt(GMM):
    """
    :param GMM: trained GMM model
    :return: get the mean of the target spectral features of each component
    """
    mid = GMM.means_.shape[1] / 2
    y_mean = GMM.means_[:, mid:]
    return y_mean

def get_cross_cov(GMM):
    # TODO use full conversion instead of VQ
    pass

def predict_GMM(src, GMM):
    """
    predict target value given src spectral features
    :param src: source spectral features
    :param GMM: trained GMM model
    :return: predicted target spectral features
    """
    m = GMM.n_components
    density_x = get_density_x(src, GMM)
    v = get_mean_tgt(GMM)
    y = np.dot(density_x,v)
    return y

if __name__=="__main__":
    pass