import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  LassoLarsIC,
                                  LassoLarsCV)
import statsmodels.api as sm


def sortnregress(X, regularisation='1-p', random_order=False):
    """ Takex n x d data, assumes order is given by increased variance,
    and regresses each node onto those with lower variance, using
    edge coefficients as structure estimates.

    regularisation:
      bic - adaptive lasso
      1-p - 1-p-vals of OLS LR (such that larger = edge)
      None / other - raw OLS LR coefficients
    """

    assert regularisation in ['bic', '1-p', 'raw', 'LassoLarsCV']

    LR = LinearRegression()
    if regularisation == 'bic':
        LL = LassoLarsIC(criterion='bic')
    elif regularisation == 'LassoLarsCV':
        LL = LassoLarsCV()

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, 0))

    if random_order:
        np.random.shuffle(increasing)

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        if regularisation == '1-p':
            ols = sm.OLS(
                X[:, target],
                sm.add_constant(X[:, covariates])).fit()
            W[covariates, target] = 1 - ols.pvalues[1:]
        elif regularisation in ['bic', 'LassoLarsCV']:
            LR.fit(X[:, covariates], X[:, target].ravel())
            weight = np.abs(LR.coef_)
            LL.fit(X[:, covariates] * weight, X[:, target].ravel())
            W[covariates, target] = LL.coef_ * weight
        elif regularisation == 'raw':
            LR.fit(X[:, covariates],
                   X[:, target].ravel())
            W[covariates, target] = LR.coef_
        else:
            raise ValueError("no such regularization")

    return W
