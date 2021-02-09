import scipy.stats
from tqdm import tqdm

from loggers import BaseLogger, PrintLogger
import collections
import numpy as np


class FeaturesPicker:
    def __init__(self, concat_matrix, logger: BaseLogger = None, size=10, identical_bar=0.6):
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default logger")
        self._size = size  # number of pairs to pick
        self._features_matrix = concat_matrix
        self._identical_bar = identical_bar  # if feature has identical values to more then bar*|V| - feature is dropped
        self._features_identicality = []  # percentage of biggest vertices group with same value per feature
        self._fill_features_identicality()
        self._best_pairs = self._pick()

    # fill best pairs with the most informative pair of features
    def _pick(self):
        raise NotImplementedError()

    def best_pairs(self):
        return self._best_pairs

    def _fill_features_identicality(self):
        self._logger.debug("start features identicality")
        rows, cols = self._features_matrix.shape
        for i in tqdm(range(cols)):
            self._features_identicality.append(collections.Counter(
                self._features_matrix[:, i].T.tolist()[0]).most_common(1)[0][1] / rows)
        self._logger.debug("end_features identicality")

    def _identicality_for(self, feature_index):
        return self._features_identicality[feature_index]

    def _is_feature_relevant(self, feature_index):
        return True if self._features_identicality[feature_index] < self._identical_bar else False


class PearsonFeaturePicker(FeaturesPicker):
    def _pick(self):
        self._logger.debug("start_pick_process")
        # the parameter which indicates how good the regression is
        rho = []
        # a list of tuples (i,j) the pairs of features we want
        best = []
        row, col = self._features_matrix.shape
        # runs over all pairs of features and check their quality parameter
        with tqdm(total=col*(col-1)//2) as t:
            for i in range(col):
                for j in range(i+1, col):
                    # obviously pair of features can't be a feature with itself..
                    t.update(1)
                    if not self._is_feature_relevant(i) or not self._is_feature_relevant(j):
                        continue
                    # returns the quality parameter (r) and how much can we rely on the result (p), for a specific pair (i,j)
                    r, p_value = scipy.stats.pearsonr(np.array(self._features_matrix[:, i]).T[0], np.array(self._features_matrix[:, j]).T[0])
                    rho.append([abs(r), i, j])

        self._logger.debug("end_pick_process")

        # higher the quality parameter the better the regression
        rho.sort(reverse=True)
        # if the user requested more pair of features than we have, very unlikely
        if self._size > len(rho):
            self._logger.error("asked for more pairs of features than there is try call pick(size=..)")
            return
        # take the features from the end (look for the highest quality parameters)
        for k in range(self._size):
            # pair is a list [quality parameter, feature i, feature j], we only need information about the features
            pair = rho[k]
            # pair[0]- quality parameter
            best.append((pair[1], pair[2], pair[0]))
            self._logger.debug("best pair:" + str(k) + "\t(" + str(pair[1]) + "," + str(pair[2]) + ")")
        return best
