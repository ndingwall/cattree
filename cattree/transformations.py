import numpy as np
from scipy.sparse import issparse
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

__author__ = "Nick Dingwall"


class OneHotDecoder(BaseEstimator, TransformerMixin):
    """
    This class reverses a one-hot encoding procedure when provided with a
    mask consisting of the one-hot encoded indices of each categorical variable
    and an outcome variable.

    The new column replaces each level of the original categorical variable
    with that level's ranked correlation with the outcome variable. For binary
    classification or regression trees and forests, encoding this way is
    guaranteed to find the optimal binary partition of levels.

    Parameters
    ----------
    masks : list
        The indices of the one-hot encoded columns of each categorical feature.
        If there is a single categorical feature, a list of ints can be passed.
        Otherwise a list of lists should be passed, with each sublist
        containing the indices of a single categorical feature's one-hot
        encoded columns.
        E.g. [[6, 7, 8], [12, 13, 14, 15]] means one categorical feature had
        three levels (in columns 6, 7 and 8 respectively), and another had 4
        levels (in columns 12 to 15).

    Attributes
    ----------
    column_mapping_to_transformed : dict
        Maps column indices from the original X to the transformed (shrunk) X'.
    column_mapping_from_transformed : dict
        Maps column indices from the transformed (shrunk) X' to the original X.
    """
    def __init__(self, masks=None):
        self.masks = masks

    # TODO add helper functions that work with DictVectorizer/OneHotEncoder

    def _format_mask(self):
        if self.masks is None or len(self.masks) == 0:
            return None

        def _check_all_ints(ls):
            return np.all([isinstance(n, int) for n in ls])

        if isinstance(self.masks[0], int):
            assert _check_all_ints(self.masks)
            return [self.masks]
        elif isinstance(self.masks[0], list) \
                and isinstance(self.masks[0][0], int):
            assert np.all([_check_all_ints(mask) for mask in self.masks])
            return self.masks
        else:
            raise ValueError(
                "'masks' must be a list of lists, or a list of ints. "
                "Instead '{}' was passed.".format(self.masks))

    @staticmethod
    def _rank_columns(A, b):
        col_sums = A.sum(axis=0)
        empty_cols = col_sums == 0
        non_empty = np.nonzero(col_sums)
        mean_outcome = np.dot(A[:, non_empty].T, b) / col_sums[non_empty]
        r = np.zeros((A.shape[1], ))
        # We want ranks to start from 1 to avoid confusing zeros here.
        ranks = np.array([rank + 1 for rank in rankdata(mean_outcome)])
        r[non_empty] = ranks
        r[empty_cols] = np.median(ranks)
        return r

    def fit(self, X, y, check_input=True):
        """Construct the mixing matrix.

        The mixing matrix converts from input arrays X to arrays X',
        where each set of one-hot encoded columns in X are represented
        by a single column in X'.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features_in]
            Not tested for sparse matrices.
            n_features_in is the number of features, including the one-hot
            encoded features, of the original data.
        y : array-like, shape = [n_samples]
            Only binary classification or regression are supported
        """
        # Checks:
        if check_input:
            X = check_array(X, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=bool)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Get an list of categorical masks and non-categorical columns
        _, self.n_features_in_ = X.shape
        masks = self._format_mask()
        masked_cols = [n for mask in masks for n in mask]
        include_singletons = [[col] for col in range(self.n_features_in_) if
                              col not in masked_cols] + self._format_mask()
        include_singletons = sorted(include_singletons, key=lambda x: x[0])

        # Make the mixing matrix
        data = []
        new_col_idx = []
        old_col_idx = []
        new_col = 0
        for mask in include_singletons:
            if len(mask) == 1:
                # A singleton mask i.e. not a one-hot encoded variable
                old_col_idx.append(mask[0])
                new_col_idx.append(new_col)
                data.append(1)
            else:
                rankings = dict(zip(mask, self._rank_columns(X[:, mask], y)))
                for old_col in mask:
                    old_col_idx.append(old_col)
                    new_col_idx.append(new_col)
                    data.append(rankings[old_col])
            new_col += 1

        self.mixing_matrix_ = csr_matrix((data, (old_col_idx, new_col_idx)))
        return self

    def transform(self, X, check_input=True):
        """
        Applies a transformation to an array.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features_in]
            Not tested for sparse matrices.
            n_features_in is the number of features, including the one-hot
            encoded features, of the original data.
        check_input : bool

        Returns
        -------
        X : array-like of shape = [n_samples, n_features_out]
            n_features_out is the number of features, where we collect all the
            levels of a categorical variable into a single feature.
        """
        check_is_fitted(self, 'mixing_matrix_')

        if check_input:
            X = check_array(X, accept_sparse="csc")
            if issparse(X):
                X.sort_indices()
                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Array has {} columns; expected {}".format(
                X.shape[1], self.n_features_in_
            ))

        return X.dot(self.mixing_matrix_.toarray())

    def old_to_new_col_index(self, old_col):
        """Returns a column index in the transformed data array corresponding
        to a provided column index in the original array."""
        check_is_fitted(self, 'mixing_matrix_')
        row = self.mixing_matrix_.toarray()[old_col, :]
        return np.where(row > 0)[0][0]

    def new_to_old_col_index(self, new_col):
        """Returns a column index in the original data array corresponding
        to a provided column index in the transformed array."""
        check_is_fitted(self, 'mixing_matrix_')
        row = self.mixing_matrix_.toarray()[:, new_col]
        return np.where(row > 0)[0]

    @property
    def column_mapping_to_transformed(self):
        check_is_fitted(self, 'mixing_matrix_')
        return {i: self.old_to_new_col_index(i)
                for i in range(self.mixing_matrix_.shape[0])}

    @property
    def column_mapping_from_transformed(self):
        check_is_fitted(self, 'mixing_matrix_')
        return {i: list(self.new_to_old_col_index(i))
                for i in range(self.mixing_matrix_.shape[1])}
