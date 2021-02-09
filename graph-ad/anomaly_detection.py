import os
import pickle
import re
import numpy as np
from features_infra.graph_features import GraphFeatures
from features_meta_ad import ANOMALY_DETECTION_FEATURES
from features_processor import FeaturesProcessor, log_norm
from loggers import PrintLogger
from temporal_graph import TemporalGraph
from tools.anomaly_picker import SimpleAnomalyPicker
from tools.beta_calculator import LinearContext, LinearMeanContext
from tools.features_picker import PearsonFeaturePicker
from tools.graph_score import KnnScore, GmmScore, LocalOutlierFactorScore, IsolationForestScore
import json
import pandas as pd


class TPGAD:
    def __init__(self, params):
        self._params = params if type(params) is dict else json.load(open(params, "rt"))
        self._logger = PrintLogger("graph-ad")
        self._temporal_graph = self._build_temporal_graph()
        self._ground_truth = self._load_ground_truth(self._params['gt']['filename'])
        self._num_anomalies = len(self._ground_truth)*2
        self._idx_to_graph = list(self._temporal_graph.graph_names())
        self._graph_to_idx = {name: idx for idx, name in enumerate(self._idx_to_graph)}

    def _load_ground_truth(self, gt_file):
        df = pd.read_csv(gt_file)
        return {self._temporal_graph.name_to_index(row.anomaly): row.get("score", 1) for i, row in df.iterrows()}

    def data_name(self):
        max_connected = "max_connected_" if self._params['features']['max_connected'] else ""
        directed = "directed" if self._params['dataset']['directed'] else "undirected"
        weighted = "weighted_" if self._params['dataset']['weight_col'] is not None else ""
        return f"{self._params['dataset']['name']}_{weighted}{max_connected}{directed}"

    def _build_temporal_graph(self):
        tg_pkl_dir = os.path.join(self._params['general']['pkl_path'], "temporal_graphs")
        tg_pkl_path = os.path.join(tg_pkl_dir, f"{self.data_name()}_tg.pkl")
        if os.path.exists(tg_pkl_path):
            self._logger.info("loading pkl file - temporal_graphs")
            tg = pickle.load(open(tg_pkl_path, "rb"))
        else:
            tg = TemporalGraph(self.data_name(), self._params['dataset']['filename'], self._params['dataset']['time_format'],
                               self._params['dataset']['time_col'], self._params['dataset']['src_col'],
                               self._params['dataset']['dst_col'],
                               weight_col=self._params['dataset'].get('weight_col', 0),
                               weeks=self._params['dataset'].get('week_split', 0),
                               days=self._params['dataset'].get('day_split', 0),
                               hours=self._params['dataset'].get('hour_split', 0),
                               minutes=self._params['dataset'].get('min_split', 0),
                               seconds=self._params['dataset'].get('sec_split', 0),
                               directed=self._params['dataset']['directed'],
                               logger=self._logger).to_multi_graph()

            tg.suspend_logger()
            if self._params['general']["dump_pkl"]:
                os.makedirs(tg_pkl_dir, exist_ok=True)
                pickle.dump(tg, open(tg_pkl_path, "wb"))
            tg.wake_logger()
        return tg

    def _calc_tg_feature_matrix(self):
        log_ext = "log_" if self._params['features']['log'] else ""
        feature_matrix_dir = os.path.join(self._params['general']['pkl_path'], "gt_feature_matrix")
        mat_pkl = os.path.join(feature_matrix_dir, f"{self.data_name()}_{log_ext}tg_feature_matrices.pkl")

        if os.path.exists(mat_pkl):
            self._logger.info("loading pkl file - graph_matrix")
            return pickle.load(open(mat_pkl, "rb"))

        gnx_to_vec = {}
        # create dir for database
        database_pkl_dir = os.path.join(self._params['general']['pkl_path'], "features", self.data_name())
        for gnx_name, gnx in zip(self._temporal_graph.graph_names(), self._temporal_graph.graphs()):
            # create dir for specific graph features
            gnx_path = os.path.join(database_pkl_dir, re.sub('[^a-zA-Z0-9]', '_', gnx_name))
            if self._params['general']["dump_pkl"]:
                os.makedirs(gnx_path, exist_ok=True)

            gnx_ftr = GraphFeatures(gnx, ANOMALY_DETECTION_FEATURES, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=self._params['features']['max_connected'])
            gnx_ftr.build(should_dump=self._params['general']["dump_pkl"],
                          force_build=self._params['general']['FORCE_REBUILD_FEATURES'])  # build features
            # calc motif ratio vector
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).as_matrix(norm_func=log_norm if self._params['features']['log'] else None)
        if self._params['general']['dump_pkl']:
            os.makedirs(feature_matrix_dir, exist_ok=True)
            pickle.dump(gnx_to_vec, open(mat_pkl, "wb"))
        return gnx_to_vec

    def _get_beta_vec(self, mx_dict, best_pairs):
        self._logger.debug("calculating beta vectors")

        if self._params['beta_vectors']['type'] == "regression":
            beta = LinearContext(self._temporal_graph, mx_dict, best_pairs,
                                 window_size=self._params['beta_vectors']['window_size'])
        elif self._params['beta_vectors']['type'] == "mean_regression":
            beta = LinearMeanContext(self._temporal_graph, mx_dict, best_pairs,
                                     window_size=self._params['beta_vectors']['window_size'])
        else:
            raise RuntimeError(f"invalid value for params[beta_vectors][type], got {self._params['beta_vectors']['type']}"
                               f" while valid options are: regression/mean_regression ")
        if self._params['general']['dump_pkl']:
            beta_pkl_dir = os.path.join(self._params['general']['pkl_path'], "beta_matrix")
            tg_pkl_path = os.path.join(beta_pkl_dir, f"{self.data_name()}_beta.pkl")
            os.makedirs(beta_pkl_dir, exist_ok=True)
            pickle.dump(beta.beta_matrix(), open(tg_pkl_path, "wb"))
        self._logger.debug("finish calculating beta vectors")

        return beta

    def _get_graphs_score(self, beta_matrix):
        score_type = self._params['score']['type']
        if score_type == "knn":
            return KnnScore(beta_matrix, self._params['score']['params']['knn']['k'], self.data_name(),
                            window_size=self._params['score']['window_size'])
        elif score_type == "gmm":
            return GmmScore(beta_matrix, self.data_name(), window_size=self._params['score']['window_size'],
                            n_components=self._params['score']['params']['gmm']['n_components'])
        elif score_type == "local_outlier":
            return LocalOutlierFactorScore(beta_matrix, self.data_name(), window_size=self._params['score']['window_size'],
                                           n_neighbors=self._params['score']['params']['local_outlier']['n_neighbors'])
        elif score_type == "isolation_forest":
            return IsolationForestScore(beta_matrix, self.data_name(), window_size=self._params['score']['window_size'],
                                        n_estimators=self._params['score']['params']['isolation_forest']['n_estimators'])
        else:
            raise RuntimeError(f"invalid value for params[beta_vectors][type], got {score_type}"
                               f" while valid options are: knn/gmm/local_outlier")

    def run_ad(self):
        mx_dict = self._calc_tg_feature_matrix()
        concat_mx = np.vstack([mx for name, mx in mx_dict.items()])
        pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params['feature_pair_picker']['num_pairs'],
                                              logger=self._logger, identical_bar=self._params['feature_pair_picker']['overlap_bar'])
        best_pairs = pearson_picker.best_pairs()
        beta_matrix = self._get_beta_vec(mx_dict, best_pairs).beta_matrix()
        scores = self._get_graphs_score(beta_matrix).score_list()

        anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, scores, self.data_name(),
                                             num_anomalies=self._num_anomalies)
        FN, TN, TP, FP, recall, precision, specificity, F1, auc, acc = anomaly_picker.build(truth=self._ground_truth)
        print("FN", FN, "|| TN", TN, "|| TP", TP, "|| FP", FP, "|| recall", recall, "|| precision", precision, "|| specificity",
              specificity, "|| F1", F1, "|| auc", auc, "|| acc", acc)
        # anomaly_picker.plot_anomalies_bokeh("", truth=self._ground_truth,
        #                                     info_text=str(self._params))
        return FN, TN, TP, FP, recall, precision, specificity, F1, auc, acc


if __name__ == "__main__":
    TPGAD("params/enron_param.json").run_ad()



