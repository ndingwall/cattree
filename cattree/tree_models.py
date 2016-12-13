import warnings

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array

from .transformations import OneHotDecoder

__author__ = "Nick Dingwall"


class CategoricalTreeBasedEstimatorBase(BaseEstimator):
    """A base class for tree-based estimators that handle categorical features.

    This class extends the sklearn tree-based models (DecisionTreeClassifier,
    DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor) to
    handle categorical variables.

    The user provides a mask consisting of a list of lists, where each sublist
    contains the indices corresponding to a single categorical variable.

    Parameters
    ----------
    underlying_model : sklearn model
        An instance of DecisionTreeClassifier, DecisionTreeRegressor,
        RandomForestClassifier or RandomForestRegressor. This should be
        hard-coded in each subclass.
    mask : list
        See OneHotDecoder for details.
    **arguments
        Parameters that will be passed to the underlying model's
        `__init__` method. Note that e.g. max_features will apply
        to the _transformed_ array that has fewer columns that the
        array passed to the `.fit` method.
    """
    def __init__(self, underlying_model, mask, **arguments):
        self.mask = mask
        if 'arguments' in arguments:
            # This check is required to deal with an idiosyncrasy of the way
            # sklearn.base.clone works, which is needed for e.g.
            # OneVsRestClassifer
            self.arguments = arguments['arguments']
        else:
            self.arguments = arguments
        self.underlying_model = underlying_model(**self.arguments)
        self.transformer = OneHotDecoder(mask)
        if isinstance(self, ClassifierMixin):
            self.y_dtypes = np.bool
        elif isinstance(self, RegressorMixin):
            self.y_dtypes = [np.float32, np.float64, np.int32, np.int64]
        else:
            raise TypeError("Subclasses should include ClassifierMixin "
                            "or RegressorMixin.")

    def fit(self, X, y, check_input=True, **fit_params):
        """Fit the tree based model.

        This preprocesses the data array using an instance of
        `OneHotDecoder` and then fits the tree-based model on its
        transformed output. In the future, this should be integrated into the
        tree-based model's `.fit` method so that an instance of
        `OneHotDecoder` is created at every node in each tree.

        Parameters
        ----------
        X : iterable
            Training data.
        y : iterable, default=None
            Training targets.
        """
        is_classification = isinstance(self, ClassifierMixin)
        if check_input:

            X = check_array(X)
            # y = check_array(y, ensure_2d=False, dtype=bool)
            y = check_array(y, ensure_2d=False,
                            dtype=self.y_dtypes)
            if is_classification:
                self.classes_ = np.unique(y)
                if len(self.classes_) > 2:
                    y = y > 0
                    warnings.warn("Multiclass classification not supported. "
                                  "Converting y to boolean.")
        Xt = self.transformer.fit_transform(X, y)
        self.underlying_model.fit(Xt, y, **fit_params)
        self._set_feature_importance()
        return self

    def predict(self, X):
        """Predict a label or regression target."""
        Xt = self.transformer.transform(X)
        return self.underlying_model.predict(Xt)

    def predict_proba(self, X):
        """Predict the probability of each binary outcome.

        Note that this method only applies to classification tasks.
        """
        # DecisionTreeClassifier needs np.float32 dtype.
        # Assume also the case for RandomForestClassifier
        Xt = self.transformer.transform(X).astype(np.float32)
        return self.underlying_model.predict_proba(Xt)

    # TODO Transform decision tree(s) structure to old variables

    def _set_feature_importance(self):
        """Transform feature importances to one-hot encoded input space.

        Since the models are trained on transformed data, but we wish to
        report importances that correspond to untransformed data, we need
        to share the importance for each categorical variable between each
        of its one-hot encoded dummies.
        """
        self.transformed_feature_importances_ = \
            self.underlying_model.feature_importances_
        self.feature_importances_ = list()
        for i, (origin_col, new_cols) in enumerate(
                self.transformer.column_mapping_from_transformed.items()):
            importance = self.transformed_feature_importances_[i]
            if len(new_cols) == 1:
                self.feature_importances_.append(importance)
            elif len(new_cols) > 1:
                n_cols = len(new_cols)
                importance /= n_cols
                for _ in new_cols:
                    self.feature_importances_.append(importance)
            else:
                raise ValueError("Somehow column_mapping_from_transformed "
                                 "got a value that isn't an int or a list: "
                                 "'{}'".format(new_cols))

    def __repr__(self):
        return "Categorical version of " + self.underlying_model.__repr__()

    def get_params(self, deep=True):
        return dict(mask=self.mask,
                    arguments=self.arguments)

    def set_params(self, **params):
        setattr(self, 'mask', params['mask'])
        setattr(self, 'arguments', params['arguments'])
        pass


class CategoricalDecisionTreeClassifier(CategoricalTreeBasedEstimatorBase,
                                        ClassifierMixin):
    """A Decision Tree Classifier that can better handle categorical features.

    This class extends the sklearn DecisionTreeClassifier.

    The user provides a mask consisting of a list of lists, where each sublist
    contains the indices corresponding to a single categorical variable.

    Parameters
    ----------
    mask : list
        See OneHotDecoder for details.
    **arguments
        All arguments are passed to the DecisionTreeClassifier's constructor.
        See `sklearn.tree.DecisionTreeClassifier` for parameters.
    """
    def __init__(self,
                 mask=None,
                 **arguments):
        super().__init__(underlying_model=DecisionTreeClassifier, mask=mask,
                         **arguments)


class CategoricalDecisionTreeRegressor(CategoricalTreeBasedEstimatorBase,
                                       RegressorMixin):
    """A Decision Tree Regressor that can better handle categorical features.

    This class extends the sklearn DecisionTreeRegressor.

    The user provides a mask consisting of a list of lists, where each sublist
    contains the indices corresponding to a single categorical variable.

    Parameters
    ----------
    mask : list
        See OneHotDecoder for details.
    **arguments
        All arguments are passed to the DecisionTreeRegressor's constructor.
        See `sklearn.tree.DecisionTreeRegressor` for parameters.
    """
    def __init__(self,
                 mask=None,
                 **arguments):
        super().__init__(underlying_model=DecisionTreeRegressor, mask=mask,
                         **arguments)


class CategoricalRandomForestClassifier(CategoricalTreeBasedEstimatorBase,
                                        ClassifierMixin):
    """A Random Forest Classifier that can better handle categorical features.

    This class extends the sklearn RandomForestClassifier.

    The user provides a mask consisting of a list of lists, where each sublist
    contains the indices corresponding to a single categorical variable.

    Parameters
    ----------
    mask : list
        See OneHotDecoder for details.
    **arguments
        All arguments are passed to the RandomForestClassifier's constructor.
        See `sklearn.tree.RandomForestClassifier` for parameters.
    """
    def __init__(self,
                 mask=None,
                 **arguments):
        super().__init__(underlying_model=RandomForestClassifier, mask=mask,
                         **arguments)


class CategoricalRandomForestRegressor(CategoricalTreeBasedEstimatorBase,
                                       RegressorMixin):
    """A Random Forest Regressor that can better handle categorical features.

    This class extends the sklearn RandomForestRegressor.

    The user provides a mask consisting of a list of lists, where each sublist
    contains the indices corresponding to a single categorical variable.

    Parameters
    ----------
    mask : list
        See OneHotDecoder for details.
    **arguments
        All arguments are passed to the RandomForestRegressor's constructor.
        See `sklearn.tree.RandomForestRegressor` for parameters.
    """
    def __init__(self,
                 mask=None,
                 **arguments):
        super().__init__(underlying_model=RandomForestRegressor, mask=mask,
                         **arguments)
