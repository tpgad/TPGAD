from sklearn import linear_model
from tqdm import tqdm
from loggers import BaseLogger, PrintLogger
import numpy as np
from multi_graph import MultiGraph


class BetaCalculator:
    def __init__(self, graphs: MultiGraph, feature_pairs, logger: BaseLogger=None):
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default graphs logger")
        self._graphs = graphs
        self._ftr_pairs = feature_pairs
        self._beta_matrix = np.zeros((self._graphs.number_of_graphs(), len(feature_pairs)))
        self._build()

    def _build(self):
        for graph_index, g_id in enumerate(self._graphs.graph_names()):
            self._beta_matrix[graph_index, :] = self._calc_beta(g_id)

    def _calc_beta(self, gid):
        raise NotImplementedError()

    def beta_matrix(self):
        return self._beta_matrix

    def to_file(self, file_name):
        out_file = open(file_name, "rw")
        for i in range(self._graphs.number_of_graphs()):
            out_file.write(self._graphs.index_to_name(i))  # graph_name
            for j in range(len(self._ftr_pairs)):
                out_file.write(str(self._beta_matrix[i][j]))  # beta_vector
            out_file.write("\n")
        out_file.close()


class LinearContext(BetaCalculator):
    def __init__(self, graphs: MultiGraph, dict_mx, feature_pairs, window_size=None):
        self._interval = int(graphs.number_of_graphs()) if window_size is None else window_size
        self._all_features = []
        self._dict_mx = dict_mx

        for graph in tqdm(graphs.graph_names()):
            m = self._dict_mx[graph]
            # self._nodes_for_graph.append(m.shape[0])
            # append graph features
            self._all_features.append(m)
            # append 0.001 for all missing nodes
            self._all_features.append(np.ones((graphs.node_count(graphs.name_to_index(graph)) - m.shape[0], m.shape[1])) * 0.001)
        # create one big matrix of everything - rows: nodes, columns: features
        self._all_features = np.concatenate(self._all_features)

        # all_ftr_graph_index - [ .... last_row_index_for_graph_i ... ]
        self._all_ftr_graph_index = np.cumsum([0] + graphs.node_count()).tolist()
        super(LinearContext, self).__init__(graphs, feature_pairs)
        self._logger.debug("finish initiating beta vectors")


    def _calc_beta(self, gid):
        beta_vec = []
        # get features matrix for interval
        g_index = self._graphs.name_to_index(gid)

        # cut the relevant part from the matrix off all features according to the interval size (and graph sizes)
        if g_index < self._interval:
            context_matrix = self._all_features[0: int(self._all_ftr_graph_index[self._interval]), :]
        else:
            context_matrix = self._all_features[int(self._all_ftr_graph_index[g_index - self._interval]):
                                                int(self._all_ftr_graph_index[g_index]), :]
        # get features matrix only for current graph
        g_matrix = self._dict_mx[gid]

        # for every one of the selected features: ftr_i, ftr_j
        # cf_window: coefficient of the linear regression on ftr_i/j in the in the window [curr - interval: current]
        # g_ftr_i: [ .... feature_i for node j .... ]
        # beta_vec = [ ....  mean < g_ftr_j - cf_window * g_ftr_i > .... ]
        for i, j, r in self._ftr_pairs:
            beta_vec.append(np.mean(g_matrix[:, j] -
                            linear_model.LinearRegression().fit(np.transpose(context_matrix[:, i].T),
                                                                np.transpose(context_matrix[:, j].T)).coef_[0][0] *
                            g_matrix[:, i]))  # * np.log(1.5 - r[0]))
        return np.asarray(beta_vec)


class LinearMeanContext(BetaCalculator):
    def __init__(self, graphs: MultiGraph, dict_mx, feature_pairs, window_size=None):
        self._interval = int(graphs.number_of_graphs()) if window_size is None else window_size
        self._dict_mx = dict_mx
        self._ordered_keys = [g_name for g_name in graphs.graph_names()]
        # all_ftr_graph_index - [ .... last_row_index_for_graph_i ... ]
        self._ref_coefficients = {}

        for i, j, r in tqdm(feature_pairs):
            self._ref_coefficients[(i, j)] = {}
            # calculate mean coefficient and beta from correlations over all context graphs
            for context_gnx_name in self._ordered_keys:
                context_gnx_mx = self._dict_mx[context_gnx_name]
                ftr_i_context_vec, ftr_j_context_vec = self._filter_ftr_vec(context_gnx_mx, i, j)
                reg_context = linear_model.LinearRegression().fit(ftr_i_context_vec, ftr_j_context_vec)
                coeff_context = reg_context.coef_.item()
                self._ref_coefficients[(i, j)][context_gnx_name] = coeff_context

        super(LinearMeanContext, self).__init__(graphs, feature_pairs)
        self._logger.debug("finish initiating beta vectors")

    def _filter_ftr_vec(self, mx, i, j, min_val=1e-9):
        min_val = np.log(min_val)
        ftr_i, ftr_j = mx[:, i], mx[:, j]
        remove_rows = []
        for row_num, (i, j) in enumerate(zip(ftr_i, ftr_j)):
            if i < min_val or j < min_val:
                remove_rows.append(row_num)
        ftr_i, ftr_j = np.delete(ftr_i, remove_rows, axis=0), np.delete(ftr_j, remove_rows, axis=0)
        if ftr_i.shape[0] == 0:
            ftr_i, ftr_j = np.vstack([ftr_i, np.matrix([[0]])]), np.vstack([ftr_j, np.matrix([[0]])])
        return ftr_i, ftr_j

    def _calc_beta(self, gid):
        beta_vec = []
        # get features matrix for interval
        g_index = self._graphs.name_to_index(gid)

        # cut the relevant part from the matrix off all features according to the interval size (and graph sizes)
        if g_index < self._interval:
            context_keys = self._ordered_keys[0: self._interval]
        else:
            context_keys = self._ordered_keys[g_index - self._interval: g_index]

        # get features matrix only for current graph
        g_matrix = self._dict_mx[gid]

        # for every one of the selected features: ftr_i, ftr_j
        # cf_window: coefficient of the linear regression on ftr_i/j in the in the window [curr - interval: current]
        # g_ftr_i: [ .... feature_i for node j .... ]
        # beta_vec = [ ....  mean < g_ftr_j - cf_window * g_ftr_i > .... ]
        for i, j, r in self._ftr_pairs:
            # calculate mean coefficient and beta from correlations over all context graphs
            sum_coeff = 0
            for context_gnx_name in context_keys:
                sum_coeff += self._ref_coefficients[(i, j)][context_gnx_name]

            # get all points + calculate average
            coeff_context = sum_coeff / self._interval

            beta_vec.append(np.mean(g_matrix[:, j] - coeff_context * g_matrix[:, i]))

        return np.asarray(beta_vec)
