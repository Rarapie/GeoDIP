"""Shared custom estimator used by the ref ML models.

The saved ``xgboost.pkl`` files were pickled with the module name
``model_classes``, so this module must be importable as a top-level package
before joblib can unpickle them.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class _XGBLabelEncoder(ClassifierMixin, BaseEstimator):
    """XGBClassifier wrapper that accepts string class labels."""

    def __init__(self, **params):
        self._params = dict(params)
        self.le = LabelEncoder()
        self.model = XGBClassifier(**self._params)

    def fit(self, X, y):
        self.le.fit(y)
        self.model.fit(X, self.le.transform(y))
        self.classes_ = np.asarray(self.le.classes_)
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return dict(self._params)

    def set_params(self, **params):
        self._params.update(params)
        self.model.set_params(**params)
        return self
