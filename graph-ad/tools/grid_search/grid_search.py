import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from itertools import product
from anomaly_detection import TPGAD


HEADER = ['dataset.name',
          'dataset.directed',
          'dataset.weight_col',
          'features.max_connected',
          'features.log',
          'feature_pair_picker.num_pairs',
          'feature_pair_picker.overlap_bar',
          'beta_vectors.type',
          'beta_vectors.window_size',
          'score.type',
          'score.window_size',
          'score.params.gmm.n_components',
          'score.params.local_outlier.n_neighbors',
          'score.params.knn.k',
          'fn', 'tn', 'tp', 'fp', 'recall', 'precision', 'specificity', 'f1'
          ]


def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value
    return dict(items())


def config_to_str(config):
    config = flatten_dict(config)
    return [str(config.get(k, "--")) for k in HEADER]


def extract_search_space(search_space, root="", search_flat_dict={}):
    for k, v in search_space.items():
        if type(v) is list:
            search_flat_dict[f"{root}/{k}"] = v
        elif type(v) is dict:
            search_flat_dict.update(extract_search_space(search_space[k], root=f"{root}/{k}",
                                                         search_flat_dict=search_flat_dict))
    return search_flat_dict


def build_config(default, config):
    new_config = deepcopy(default)
    for key_path, v in config.items():
        curr_dict = new_config
        key_path = key_path[1:].split("/")
        for k in key_path[:-1]:
            if k not in curr_dict:
                curr_dict[k] = {}
            curr_dict = curr_dict[k]
        curr_dict[key_path[-1]] = v

    irrelevant_score_types = ['gmm', 'knn', "local_outlier"]
    irrelevant_score_types.remove(new_config['score']['type'])
    for k in irrelevant_score_types:
        if k in new_config['score']['params']:
            del new_config['score']['params'][k]

    sha = hashlib.md5(str(new_config).encode()).hexdigest()
    return new_config, sha


def config_iterator(default_params, search_space):
    default_params = default_params if type(default_params) is dict else json.load(open(default_params, "rt"))
    search_space = search_space if type(search_space) is dict else json.load(open(search_space, "rt"))

    overlap_configurations = set()
    config_dict = extract_search_space(search_space)
    sorted_keys = [k for k in sorted(config_dict)]
    all_configs = list(product(*[config_dict[k] for k in sorted_keys]))
    for config in all_configs:
        config, sha = build_config(default_params, {k: v for k, v in zip(sorted_keys, config)})
        if sha in overlap_configurations:
            continue
        else:
            overlap_configurations.add(sha)
            yield config


def run_grid(default_params, search_space, res_dir):
    now = datetime.now().strftime("%d%m%y_%H%M%S")
    default_params = default_params if type(default_params) is dict else json.load(open(default_params, "rt"))
    res_filename = os.path.join(res_dir, f"{default_params['dataset']['name']}_grid_{now}.csv")
    out = open(res_filename, "wt")
    out.write(f"{','.join(HEADER)}\n")
    for config in config_iterator(default_params, search_space):
        tpgad = TPGAD(config)
        fn, tn, tp, fp, recall, precision, specificity, f1, auc, acc = tpgad.run_ad()
        table_row = config_to_str(config)
        table_row[HEADER.index('fn')] = str(fn)
        table_row[HEADER.index('tn')] = str(tn)
        table_row[HEADER.index('tp')] = str(tp)
        table_row[HEADER.index('fp')] = str(fp)
        table_row[HEADER.index('recall')] = str(recall)
        table_row[HEADER.index('precision')] = str(precision)
        table_row[HEADER.index('specificity')] = str(specificity)
        table_row[HEADER.index('f1')] = str(f1)
        table_row[HEADER.index('AUC')] = str(auc)
        table_row[HEADER.index('accuracy')] = str(acc)
        out.write(f"{','.join(table_row)}\n")


if __name__ == '__main__':
    default_params = "C:\\Users\\kfirs\\lab\\TPGAD\\graph-ad\\params\\secrepo_param.json"
    search_space = "C:\\Users\\kfirs\\lab\\TPGAD\\graph-ad\\tools\\grid_search\\search_space\\search_space_params.json"
    results_dir = "C:\\Users\\kfirs\\lab\\TPGAD\\graph-ad\\tools\\grid_search\\grid_results"
    run_grid(default_params, search_space, results_dir)
