import sklearn.mixture
import numpy as np

def train_GMM(jnt):
    """
    train GMM with joint vectors
    :param jnt: joint vector of spectral features
    :return: trained GMM model
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=40, covariance_type='full')
    GMM.fit(jnt)
    return GMM

def get_density_x(src, GMM):
    """
    :param src: source spectral features
    :param GMM: trained GMM model
    :return: get the conditional probability that src belong to a component
    """
    mid = GMM.means_.shape[1] / 2
    x_GMM = sklearn.mixture.GaussianMixture(n_components=40, covariance_type='full')
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
    mid = GMM.means_.shape[1] / 2
    return GMM.covariances_[:, mid:, :mid]

def get_xx_cov(GMM):
    mid = GMM.means_.shape[1] / 2
    return GMM.covariances_[:, :mid, :mid]

def get_x_mean(GMM):
    mid = GMM.means_.shape[1] / 2
    return GMM.means_[:, 0:mid]

def predict_GMM_VQ(src, GMM):
    """
    predict target value given src spectral features by VQ conversion
    :param src: source spectral features
    :param GMM: trained GMM model
    :return: predicted target spectral features
    """
    density_x = get_density_x(src, GMM)
    v = get_mean_tgt(GMM)
    m = GMM.n_components
    T, n_mcep = src.shape
    y = np.zeros((T, n_mcep))

    for t in range(T):
        for i in range(m):
            y[t] = y[t] + np.dot(density_x[t][i], v[i])

    return y

def predict_GMM_FULL(src, GMM):
    """
    predict target value given src spectral features by Full conversion
    :param src: source spectral features
    :param GMM: trained GMM model
    :return: predicted target spectral features
    """
    density_x = get_density_x(src, GMM)
    v = get_mean_tgt(GMM)
    diag = get_cross_cov(GMM)
    sig = get_xx_cov(GMM)
    mean_x = get_x_mean(GMM)
    m = GMM.n_components
    T, n_mcep = src.shape
    y = np.zeros((T, n_mcep))
    A = np.zeros((m,n_mcep,n_mcep))

    for i in range(m):
        A[i] = np.dot(diag[i], np.linalg.inv(sig[i]))
    for t in range(T):
        for i in range(m):
            tmp = np.dot(A[i], (src[t] - mean_x[i]))
            y[t] = y[t] + np.dot(density_x[t][i], v[i] + tmp)

    return y

if __name__=="__main__":
    pass