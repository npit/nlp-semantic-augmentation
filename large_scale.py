from os.path import join, exists, isabs, basename, splitext
import argparse
from os import makedirs
import subprocess
from collections import OrderedDict
import yaml
import pickle
import pandas as pd
from functools import reduce
import itertools
from copy import deepcopy
from numpy import round
import smtplib
import logging
import os
import getpass
from utils import info, error, datetime_str, as_list, setup_simple_logging, ordered_load, ordered_dump, warning

"""Script to produce large-scale semantic neural augmentation experiments

The experiment variation parameters should be in a "params" field in the configuration file.
Values in a list are interpreted as different parameters (so for list literal values, add an additional list wrap)
"""

class VariableConf(OrderedDict):
    id = None

    @staticmethod
    def get_copy(instance):
        new_instance = deepcopy(OrderedDict(instance))
        new_instance = VariableConf(new_instance)
        new_instance.id = instance.id
        return new_instance

    def __init__(self, existing=None):
        if existing is not None:
            error("Ordered dict required for variable config. ", type(existing) != OrderedDict)
            super().__init__(existing)
        else:
            super().__init__()
        self.id = ""

    def add_variable(self, keys, value):
        info("Setting variable field: {} / value: {} -- current conf id: {}".format(keys, value, self.id))
        conf = self
        for k, key in enumerate(keys[:-1]):
            if key not in conf:
                error("Key not present in configuration and it's not a parent component name.", k != len(keys) -2)
                conf[key] = {}
            conf = conf[key]
        if  keys[-1] in conf:
            error("Variable key already in configuration!")
        conf[keys[-1]] = value

        # use the last key for the id -- revisit if there's ambiguity
        # self.id = "_".join(keys) + "_" + str(value)
        if type(value) == list:
            strvalue = "_".join(map(str, value))
        else:
            strvalue = str(value)
        strvalue = strvalue.replace("/", "_")

        if len(self.id) > 0:
            self.id += "_"
        self.id += keys[-1] + "_" + strvalue
        info("Final conf id: {}".format(self.id))

    def __str__(self):
        return self.id + " : " + super().__str__()


exlogger = logging.getLogger("experiments")

def sendmail(mail, passw, msg, title="nle"):
    # email me
    TO = mail
    SUBJECT = title
    TEXT = msg
    # Gmail Sign In
    gmail_sender = mail
    recipient = mail
    gmail_passwd = passw

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join(['To: %s' % TO, 'From: %s' % gmail_sender, 'Subject: %s' % SUBJECT, '', TEXT])
    try:
        server.sendmail(gmail_sender, [TO], BODY)
        info('Email sent to [%s]' % recipient)
    except Exception as x:
        info('Error sending mail to [%s]' % recipient)
        error(x)

def traverse_dict(ddict, key, prev_keys):
    res = []
    if key is None:
        for key in ddict:
            rres = traverse_dict(ddict, key, prev_keys)
            res.append(rres)
        return res
    if type(ddict[key]) is dict:
        prev_keys.append(key)
        res = traverse_dict(ddict[key], None, prev_keys)
    else:
        val = ddict[key]
        if type(val) is not list:
            val = [val]
        res = (val, prev_keys + [key])
    return res

def keyseq_exists(key_seq, ddict):
    # make sure no key sequence param exists in the base config
    for key in key_seq:
        try:
            ddict = ddict[key]
        except KeyError:
            return False
    return True

def parse_variable_component(var):
    variable_name = var["name"]
    variable_components = var["values"]
    error("Variable components of length 1, for name {}".format(variable_name), len(variable_components) == 1)

def expand_configs(configs, keys, values):
    info("Propagating values {} for field: {}".format(values, keys))
    num_values = len(values)
    if len(configs) > 0:
        info("Current config ids: {}".format([c.id for c in configs]))
    new_configs = []
    for _ in range(num_values -1):
        for conf in configs:
            new_configs.append(VariableConf.get_copy(conf))
    configs += new_configs

    num_per_value = len(configs) / num_values
    # partition into num_values
    conf_idx, value_idx, assignment_count = 0, 0, 0
    while conf_idx < len(configs):
        if assignment_count == num_per_value:
            value_idx += 1
            assignment_count = 0
        conf = configs[conf_idx]
        conf.add_variable(keys, values[value_idx])
        assignment_count += 1
        conf_idx += 1
    error("Value index error after expansion: {} but values are {}".format(value_idx, len(values)), value_idx != len(values) -1)

def populate_configs(configs, chain_name, component_name, field_name, field_value):
    # info("Setting regular field: {} {} {} {}".format(chain_name, component_name, field_name, field_value))
    for conf in configs:
        if chain_name not in conf["chains"]:
            conf["chains"][chain_name] = {}
        if component_name not in conf["chains"][chain_name]:
            conf["chains"][chain_name][component_name] = {}
        if field_value is None:
            conf["chains"][chain_name][component_name] = field_name
        else:
            conf["chains"][chain_name][component_name][field_name] = field_value

def make_configs(base_config, run_dir, sources_dir="./"):
    # read chains
    configs = [VariableConf()]
    configs[0]["chains"] = {}

    chains = base_config["chains"]
    for chain_name, chain_body in chains.items():
        print(OrderedDict(configs[0]))
        for component_name, component_body in chain_body.items():
            if type(component_body) not in [dict, OrderedDict]:
                populate_configs(configs, chain_name, component_name, component_body, None)
                continue
            for  field_name, field_value in component_body.items():
                if field_name == "variables":
                    for variable_field_name, variable_field_values in component_body[field_name].items():
                        # expand configurations
                        expand_configs(configs, ["chains", chain_name, component_name, variable_field_name], variable_field_values)
                else:
                    populate_configs(configs, chain_name, component_name, field_name, field_value)

    info("Expansion resulted in {} configurations.".format(len(configs)))
    # parsed chains -- copy rest of values
    for key in base_config:
        if key in ["experiments", "chains"]:
            continue
        for conf in configs:
            conf[key] = deepcopy(base_config[key])

    # set misc
    for conf in configs:
        conf["folders"]["run"] = join(run_dir, conf.id)
        # cases for explicit serialization and raw data folders
        ser_folder, raw_folder = conf["folders"]["serialization"], conf["folders"]["raw_data"]
        if not isabs(ser_folder):
            conf["folders"]["serialization"] = join(sources_dir, ser_folder)
        if not isabs(raw_folder):
            conf["folders"]["raw_data"] = join(sources_dir, raw_folder)
        conf["misc"]["run_id"] = conf.id
    return sorted(configs, key=lambda x: x.id)

# make a run id name out of a list of nested dict keys and a configuration dict
def make_run_ids(keychains, confs):
    names = []
    for conf in confs:
        name_components = []
        for keychain in keychains:
            name_components.append(reduce(dict.get, keychain, conf))
        names.append("_".join(map(str, name_components)))
    return names

# get a nested dict value from a list of keys
def get_kseq_value(kseq, ddict):
    res = ddict
    for k in kseq:
        res = res[k]
    return res

def filter_testing(configs, config_file):
    # discard configurations with incompatible components
    out_conf = []
    # read bad combos
    with open(join(config_file + ".bad_combos")) as f:
        bad_combos_lists = yaml.load(f, Loader=yaml.SafeLoader)
    for conf in zip(configs):
        bad_conf = False
        for bad_combo in bad_combos_lists:
            # if all bad key-value pairs exist in the conf, drop it
            combo_components_exist = [
                    keyseq_exists(keyseq, conf) and value == get_kseq_value(keyseq, conf) if type(value) is not list else get_kseq_value(keyseq, conf) in value \
                    for (keyseq, value) in bad_combo]
            if all(combo_components_exist):
                bad_conf = True
                info("Omitting incompatible config {} with bad entries: {}".format(conf.id, bad_combo))
                break
        if not bad_conf:
            out_conf.append(conf)
    return out_conf

def main(config_file="chain.large.config.yml", is_testing_run=False):
    # settable parameters
    ############################################################

    email = "pittarasnikif@gmail.com"
    passw = None

    ############################################################

    # set the expeirment parameters via a configuration list
    with open(config_file) as f:
        conf = ordered_load(f, Loader=yaml.SafeLoader)

    exps = conf["experiments"]

    # folder to run experiments in
    run_dir = exps["run_folder"]

    # dir checks
    # ----------
    # virtualenv folder
    venv_dir = conf["experiments"]["venv"] if "venv" in conf["experiments"] else None
    # results csv file
    # results_file = conf["experiments"]["results_file"]
    results_file = join(run_dir, "run_results.csv")

    if venv_dir and not exists(venv_dir):
        error("Virtualenv dir {} not found".format(venv_dir))
    if not exists(run_dir):
        info("Run dir {} not found, creating.".format(run_dir))
        makedirs(run_dir)

    # logging
    os.makedirs(run_dir, exist_ok=True)
    setup_simple_logging(conf["print"]["log_level"], logging_dir=run_dir)

    info("Generating configurations from source file {}".format(config_file))

    # evaluation measures
    eval_measures = as_list(exps["measures"]) if "measures" in exps else ["f1-score", "accuracy"]
    aggr_measures = as_list(exps["label_aggregation"]) if "label_aggregation" in exps else ["macro", "micro"]
    stat_functions = as_list(exps["fold_aggregation"]) if "fold_aggregation" in exps else ["mean"]
    run_types = as_list(exps["run_types"]) if "run_types" in exps else ["run"]

    # folder where run scripts are
    sources_dir = exps["sources_dir"] if "sources_dir" in exps else os.getcwd()
    warning("Defaulting sources folder to the current directory: {}".format(sources_dir))
    error("Main module: {} not found. Is the sources dir ok?".format(join(sources_dir, "main.py")), not exists(join(sources_dir, "main.py")))

    configs = make_configs(conf, run_dir, sources_dir)
    # check run id uniqueness
    if len(set([c.id for c in configs])) != len(configs):
        error("Duplicate run folders from the input: {}".format([c.id for c in configs]))
    if len(set([c['folders']['run'] for c in configs])) != len(configs):
        error("Duplicate run folders from the input: {}".format([c["folders"]["run"] for c in configs]))
    # if we're running a testing suite, filter out incompatible configs
    if is_testing_run:
        configs = filter_testing(configs, config_file)

    # mail
    do_send_mail = exps["send_mail"] if "send_mail" in exps else None
    if do_send_mail:
        passw = getpass.getpass()

    # copy the configuration file in the target directory
    copied_conf = join(run_dir, basename(config_file))
    if exists(copied_conf):
        config_to_copy = OrderedDict({k: v for (k, v) in conf.items() if k != "experiments"})
        # make sure it's the same effing config
        with open(copied_conf) as f:
            cconf = ordered_load(f, Loader=yaml.SafeLoader)
        cconf = OrderedDict({k: v for (k, v) in cconf.items() if k != "experiments"})
        if config_to_copy != cconf:
            print(config_to_copy, cconf)
            print([x.keys() for x in [config_to_copy, cconf]])
            error("The workflow contents derived from the original config [{}] differ from the ones in the experiment directory: [{}]!".format(config_file, copied_conf))
    else:
        info("Copying experiments configuration at {}".format(copied_conf))
        with open(copied_conf, "w") as f:
            ordered_dump(OrderedDict(conf), f)

    results = {}

    #################################################################################

    # prelim experiments
    for conf_index, conf in enumerate(configs):
        run_id = conf.id
        info("Running experimens for configuration {}/{}: {}".format(conf_index + 1, len(configs), run_id))
        experiment_dir = conf["folders"]["run"]
        completed_file = join(experiment_dir, "completed")
        error_file = join(experiment_dir, "error")
        # results to run folders, if not specified otherwise
        respath = join(experiment_dir, "results")
        if not isabs(respath):
            conf["folders"]["results"] = join(experiment_dir, respath)

        if exists(completed_file):
            info("Skipping completed experiment {}".format(run_id))
        else:
            # run it
            if exists(error_file):
                os.remove(error_file)
            makedirs(experiment_dir, exist_ok=True)

            conf_path = join(experiment_dir, "config.yml")
            with open(conf_path, "w") as f:
                ordered_dump(OrderedDict(conf), f)
            info("Configuration file: {}".format(conf_path))
            # write the run script file
            script_path = join(experiment_dir, "run.sh")
            with open(script_path, "w") as f:
                if venv_dir:
                    f.write("source \"{}/bin/activate\"".format(venv_dir))
                f.write("cd \"{}\"\n".format(sources_dir))
                f.write("python3 \"{}\" \"{}\" && touch \"{}\" && exit 0\n".format(join(sources_dir, "main.py"), conf_path, completed_file))
                f.write("touch '{}' && exit 1\n".format(error_file))

            subprocess.run(["/usr/bin/env", "bash", script_path])
            if exists(error_file):
                print("An error has occurred in the run, exiting.")
                info("An error has occurred in the run, exiting.")
                if do_send_mail:
                    sendmail(email, passw, "an error occurred")
                exit(1)
        # read experiment results
        exp_res_file = join(experiment_dir, "results", "results.pickle")
        with open(exp_res_file, "rb") as f:
            res_data = pickle.load(f)["results"]
        results[run_id] = res_data

    # messages = []
    total_results = {}
    # show results
    for stat in stat_functions:
        print("Results regarding {} statistic:".format(stat))
        print_vals = {}
        for run_id in results:
            print_vals[run_id] = {}
            for m in eval_measures:
                for run in run_types:
                    for ag in aggr_measures:
                        if ag not in results[run_id].loc[m][run]:
                            # messages.append("Run {}: Aggregation {} incompatible with measure {}.".format(run_id, ag, m))
                            continue
                        header = "{}.{}.{}.{}".format(run[:3], m[:3], ag[:3], stat)
                        if stat == "var":
                            val = round(results[run_id].loc[m][run][ag]["var"], decimals=4)
                        elif stat == "mean":
                            val = round(results[run_id].loc[m][run][ag]["mean"], decimals=4)
                        elif stat == "std":
                            val = round(results[run_id].loc[m][run][ag]["std"], decimals=4)
                        print_vals[run_id][header] = val
        # print'em
        info("SCORES:")
        df = pd.DataFrame.from_dict(print_vals, orient='index')
        print(df.to_string())


        info("RANKS:")
        ranked = pd.concat([df[c].rank(ascending=False) for c in [x for x in df.columns if x.startswith('run')]], axis=1)
        avg_rank = sum([ranked[x].values for x in ranked.columns])
        ranked['avg rank'] = pd.Series(avg_rank, index=ranked.index)
        print(ranked.to_string())

        total_results[stat] = print_vals
    info("Writing these results to file {}".format(results_file))
    total_df = pd.DataFrame.from_dict(total_results, orient='index')
    total_df.to_csv(results_file)

    # [info(msg) for msg in messages]
    if do_send_mail:
        sendmail(email, passw, "run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Configuration .yml file for the run.", nargs="?", default="large.config.yml")
    args = parser.parse_args()
    # main(args.config_file)
    main("multiling.large.config.yml")
