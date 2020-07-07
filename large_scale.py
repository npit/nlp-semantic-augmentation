""" Large-scale experiment runner module for smaug """
import argparse
import getpass
import itertools
import logging
import os
import pickle
import shutil
import subprocess
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from itertools import product
from os import listdir, makedirs
from os.path import basename, dirname, exists, isabs, isdir, join
from shutil import rmtree

import yaml
from numpy import round

import pandas as pd
import stattests
from experiments.utils import (compare_dicts, filter_testing, keyseq_exists,
                               sendmail)
from experiments.variable_config import VariableConf
from stattests import instantiator
from utils import (as_list, datetime_str, error, info, write_ordered_dump,
                   read_ordered_yaml, setup_simple_logging, warning)


# bla bloo
"""Script to produce large-scale semantic neural augmentation experiments

The experiment variation parameters should be in a "params" field in the configuration file.
Values in a list are interpreted as different parameters (so for list literal values, add an additional list wrap)
"""



VARIABLE_KEY_NAME = "variable"
EXPERIMENTS_KEY_NAME = "experiments"

exlogger = logging.getLogger(EXPERIMENTS_KEY_NAME)


def expand_configs(configs, keys, values):
    info("Propagating values {} for field: {}".format(values, keys))
    num_values = len(values)
    if len(configs) > 0:
        info("Current config ids: {}".format([c.id for c in configs]))
    new_configs = []
    for _ in range(num_values - 1):
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
    error(
        "Value index error after expansion: {} but values are {}".format(
            value_idx, len(values)), value_idx != len(values) - 1)


def populate_configs(configs, chain_name, component_name, field_name,
                     field_value):
    # info("Setting regular field: {} {} {} {}".format(chain_name, component_name, field_name, field_value))
    for conf in configs:
        if chain_name not in conf["chains"]:
            conf["chains"][chain_name] = {}
        if component_name not in conf["chains"][chain_name]:
            conf["chains"][chain_name][component_name] = {}
        if field_value is None:
            conf["chains"][chain_name][component_name] = field_name
        else:
            conf["chains"][chain_name][component_name][
                field_name] = field_value


def make_configs(base_config, run_dir, sources_dir="./"):
    """Produce multiple configuration objects from a single configuration with variable fields"""
    # read chains
    configs = [VariableConf()]
    configs[0]["chains"] = {}

    chains = base_config["chains"]
    for chain_name, chain_body in chains.items():

        for component_name, component_body in chain_body.items():
            if type(component_body) not in [dict, OrderedDict]:
                populate_configs(configs, chain_name, component_name,
                                 component_body, None)
                continue
            for field_name, field_value in component_body.items():
                if field_name == VARIABLE_KEY_NAME:

                    for variable_field_name, variable_field_values in component_body[
                            field_name].items():
                        # expand configurations
                        expand_configs(configs, [
                            "chains", chain_name, component_name,
                            variable_field_name
                        ], variable_field_values)
                else:
                    populate_configs(configs, chain_name, component_name,
                                     field_name, field_value)

    info("Expansion resulted in {} configurations.".format(len(configs)))
    # parsed chains -- copy rest of values
    for key in base_config:
        if key in [EXPERIMENTS_KEY_NAME, "chains"]:
            continue
        for conf in configs:
            conf[key] = deepcopy(base_config[key])

    # set misc
    for conf in configs:
        conf["folders"]["run"] = join(run_dir, conf.id)
        # cases for explicit serialization and raw data folders
        ser_folder, raw_folder = conf["folders"]["serialization"], conf[
            "folders"]["raw_data"]
        if not isabs(ser_folder):
            conf["folders"]["serialization"] = join(sources_dir, ser_folder)
        if not isabs(raw_folder):
            conf["folders"]["raw_data"] = join(sources_dir, raw_folder)
        conf["misc"]["run_id"] = conf.id
    return sorted(configs, key=lambda x: x.id)


def print_existing_csv_results(path):
    info("Printing existing results from {}".format(path))
    info("Showing mean aggregation")
    x = pd.read_csv(path, index_col=0).transpose()['mean']
    dct = {}
    for i in x.index:
        dct[i.replace('dimension', 'dim')] = eval(x[i])
    print_dataframe_results(dct)


def print_dataframe_results(dict_scores):
    if not dict_scores:
        print("<empty scores!>")
        return
    df = pd.DataFrame.from_dict(dict_scores, orient='index')
    # print'em
    info("SCORES:")
    print(df.to_string())
    info("RANKS:")
    ranked = pd.concat([
        df[c].rank(ascending=False)
        for c in [x for x in df.columns if x.startswith('run')]
    ],
                       axis=1)
    avg_rank = sum([ranked[x].values for x in ranked.columns])
    ranked['avg rank'] = pd.Series(avg_rank, index=ranked.index)
    print(ranked.to_string())


def main(input_path, only_report=False, force_dir=False, no_config_check=False, restart=False, is_testing_run=False):
    # settable parameters
    ############################################################

    email = "pittarasnikif@gmail.com"
    passw = None

    ############################################################

    # set the experiment parameters
    error("Non-existent input path: {} ".format(input_path),
          not exists(input_path))
    if isdir(input_path):
        # assume a single .yml file in the directory
        ymls = [
            x for x in listdir(input_path) if any(
                x.endswith(suff) for suff in [".yaml", ".yml"])
        ]
        error(
            "Input path {} is a directory with no yaml configuration files.".
            format(input_path), not ymls)
        error(
            "Input path is a directory with more than one yaml configuration files."
            .format(input_path),
            len(ymls) > 1)
        config_file = join(input_path, ymls[0])
    else:
        config_file = input_path

    # if input file is existing csv scores, just print them
    if config_file.endswith(".csv"):
        print_existing_csv_results(config_file)
        return

    conf = read_ordered_yaml(config_file)

    try:
        exps = conf[EXPERIMENTS_KEY_NAME]
    except KeyError:
        error(f"Need an [{EXPERIMENTS_KEY_NAME}] key for large-scale experiments.")

    # folder to run experiments in
    run_dir = exps["run_folder"]
    if force_dir:
        warning(
            "Overriding experiment folder from yml value: {} to current dir: {}, due to force-dir"
            .format(run_dir, dirname(run_dir)))
        run_dir = dirname(input_path)
    if not isabs(run_dir):
        run_dir = join(os.getcwd(), run_dir)

    # dir checks
    # ----------
    # virtualenv folder
    venv_dir = conf[EXPERIMENTS_KEY_NAME]["venv"] if "venv" in conf[
        EXPERIMENTS_KEY_NAME] else None
    # results csv file
    # results_file = conf["experiments"]["results_file"]
    results_file = join(run_dir, "run_results.csv")

    if venv_dir and not exists(venv_dir):
        error("Virtualenv dir {} not found".format(venv_dir))
    if not exists(run_dir):
        info("Run dir {} not found, creating.".format(run_dir))
        makedirs(run_dir)
    else:
        error("Specified a non-dir path as the running directory: {}".format(run_dir), not isdir(run_dir))
        if restart:
            warning("Specified restart, and experiment dir {} exists. Deleting!")
            rmtree(run_dir)
            makedirs(run_dir)

    # logging
    os.makedirs(run_dir, exist_ok=True)
    setup_simple_logging(conf["print"]["log_level"], logging_dir=run_dir)

    info("Generating configurations from source file {}".format(config_file))

    # evaluation measures
    try:
        eval_measures = as_list(
            exps["measures"]) if "measures" in exps else ["f1-score", "accuracy"]
        print(eval_measures)
        aggr_measures = as_list(exps["label_aggregation"]) if "label_aggregation" in exps \
            else ["macro", "micro"]
        stat_functions = as_list(
            exps["fold_aggregation"]) if "fold_aggregation" in exps else ["mean"]
        run_types = as_list(exps["run_types"]) if "run_types" in exps else ["run"]
        do_sstests = "sstests" in exps
        if not do_sstests:
            warning("No statistical tests specified.")
        else:
            sstests = ["tukeyhsd"] if "names" not in exps["sstests"] else as_list(exps["sstests"]["names"])
            sstests_measures = ["f1-score"] if "measures" not in exps["sstests"] else as_list(exps["sstests"]["measures"])
            sstests_aggregations = ["macro"] if "aggregations" not in exps["sstests"] else as_list(exps["sstests"]["aggregations"])
            sstests_limit_vars = None if "limit_variables" not in exps["sstests"] else as_list(exps["sstests"]["limit_variables"])
    except Exception as ex:
        error("Failed to read evaluation / testing options due to: [{}]".format(ex))

    # folder where run scripts are
    sources_dir = exps["sources_dir"] if "sources_dir" in exps else os.getcwd()
    warning("Defaulting sources folder to the current directory: {}".format(
        sources_dir))
    error(
        "Main module: {} not found. Is the sources dir ok?".format(
            join(sources_dir, "main.py")),
        not exists(join(sources_dir, "main.py")))

    configs = make_configs(conf, run_dir, sources_dir)
    # check run id uniqueness
    if len(set([c.id for c in configs])) != len(configs):
        error("Duplicate run folders from the input: {}".format(
            [c.id for c in configs]))
    if len(set([c['folders']['run'] for c in configs])) != len(configs):
        error("Duplicate run folders from the input: {}".format(
            [c["folders"]["run"] for c in configs]))
    # if we're running a testing suite, filter out incompatible configs
    if is_testing_run:
        configs = filter_testing(configs, config_file)

    # mail
    do_send_mail = exps["send_mail"] if "send_mail" in exps else None
    if do_send_mail:
        passw = getpass.getpass()

    # copy the experiments configuration file in the target directory
    experiments_conf_path = join(run_dir, basename(config_file))
    if exists(experiments_conf_path):
        # make sure it's the same effing config, unless check is overriden
        if not no_config_check:
            config_to_copy = OrderedDict(
                {k: v
                for (k, v) in conf.items() if k != EXPERIMENTS_KEY_NAME})
            existing_exp_conf = read_ordered_yaml(experiments_conf_path)
            existing_exp_conf = OrderedDict({
                k: v for (k, v) in existing_exp_conf.items() if k != EXPERIMENTS_KEY_NAME
            })
            equal, diff = compare_dicts(config_to_copy, existing_exp_conf)
            if not equal:
                error(
                    "The workflow contents derived from the original config [{}] differ from the ones in the experiment directory: [{}]!\nDifference is: {}"
                    .format(config_file, experiments_conf_path, diff))
    else:
        if not only_report:
            info("Copying experiments configuration at {}".format(
                experiments_conf_path))
            with open(experiments_conf_path, "w") as f:
                write_ordered_dump(OrderedDict(conf), f)
        else:
            info(
                "Only-report run: will not copy experiment configuration at {}"
                .format(experiments_conf_path))

    results, result_paths = {}, {}

    #################################################################################
    skipped_configs = []

    # prelim experiments
    for conf_index, conf in enumerate(configs):
        run_id = conf.id
        info("Running experimens for configuration {}/{}: {}".format(
            conf_index + 1, len(configs), run_id))
        experiment_dir = conf["folders"]["run"]
        completed_file = join(experiment_dir, "completed")
        error_file = join(experiment_dir, "error")
        # results to run folders, if not specified otherwise
        respath = join(experiment_dir, "results")
        if not isabs(respath):
            conf["folders"]["results"] = join(experiment_dir, respath)

        if exists(completed_file):
            info("Skipping completed experiment {}".format(run_id))
        elif only_report:
            info("Only-report execution: skipping non-completed experiment {}".
                 format(run_id))
            skipped_configs.append(run_id)
            continue
        else:
            # run it
            if exists(error_file):
                os.remove(error_file)
            makedirs(experiment_dir, exist_ok=True)

            conf_path = join(experiment_dir, "config.yml")
            if exists(conf_path) and not no_config_check:
                warning("Configuration file at {} already exists!".format(
                    conf_path))
                existing = read_ordered_yaml(conf_path)
                equal, diff = compare_dicts(existing, conf)
                if not equal:
                    error(
                        "Different local config encountered: {} \nDifference: {}"
                        .format(conf_path, diff))
                #if not (OrderedDict(conf) == existing):
                #    error("Different local config encountered at {}".format(conf_path))
            else:
                with open(conf_path, "w") as f:
                    write_ordered_dump(OrderedDict(conf), f)
            info("Configuration file: {}".format(conf_path))
            # write the run script file
            script_path = join(experiment_dir, "run.sh")
            with open(script_path, "w") as f:
                if venv_dir:
                    f.write("source \"{}/bin/activate\"".format(venv_dir))
                f.write("cd \"{}\"\n".format(sources_dir))
                f.write(
                    "python3 \"{}\" \"{}\" && touch \"{}\" && exit 0\n".format(
                        join(sources_dir, "main.py"), conf_path,
                        completed_file))
                f.write("touch '{}' && exit 1\n".format(error_file))

            subprocess.run(["/usr/bin/env", "bash", script_path])
            if exists(error_file):
                print("An error has occurred in the run, exiting.")
                info("An error has occurred in the run, exiting.")
                if do_send_mail:
                    sendmail(email, passw, "an error occurred")
                exit(1)
        # read experiment results
        exp_res_file = join(experiment_dir, "results", "results.pkl")
        with open(exp_res_file, "rb") as f:
            res_data = pickle.load(f)
        results[run_id] = res_data
        result_paths[run_id] = exp_res_file

    # messages = []
    total_results = {}
 
    # show results
    for stat in stat_functions:
        info("Results regarding {} statistic:".format(stat))
        print_vals = {}
        for run_id in results:
            print_vals[run_id] = {}
            for m in eval_measures:
                for run in run_types:
                    for ag in aggr_measures:
                        try:
                            results[run_id][run][m][ag]
                        except KeyError:
                            continue
                        header = "{}.{}.{}.{}".format(run[:3], m[:3], ag[:3],
                                                      stat)

                        if stat in "var mean std".split():
                            val = results[run_id][run][m][ag][stat]
                        if val is None:
                            continue
                        val = round(val, decimals=4)
                        print_vals[run_id][header] = val
        # print'em
        info("SCORES:")
        print_dataframe_results(print_vals)

        total_results[stat] = print_vals
    info("Writing these results to file {}".format(results_file))
    total_df = pd.DataFrame.from_dict(total_results, orient='index')
    if total_df.size == 0:
        info("No results parsed.")
    else:
        total_df.to_csv(results_file)

    if skipped_configs:
        for s, sk in enumerate(skipped_configs):
            info("Skipped incomplete config: {}/{} : {}".format(
                s + 1, len(skipped_configs), sk))

    if do_sstests:
        do_stat_sig_testing(sstests, sstests_measures, sstests_aggregations, configs, results, sstests_limit_vars)

    # [info(msg) for msg in messages]
    if do_send_mail:
        sendmail(email, passw, "run complete.")

def do_stat_sig_testing(methods, measures, label_aggregations, configs, results, limit_variables=None, run_mode="run"):
    testable_variables = list(configs[0].ddict.keys())
    if limit_variables:
        testable_variables = [x for x in testable_variables if x in limit_variables]
    info("Running statistical tests on{} variables: {}".format(" all" if limit_variables is None else " specified", testable_variables))
    for method, measure, label_aggregation in product(methods, measures, label_aggregations):
        info("Running statistical testing via {} on {} {}".format(method, label_aggregation, measure))
        # get ids and variable values per configuration
        df_inputs = []
        try:
            for run_id in results:
                # find corresp. configuration
                conf = [c for c in configs if c.id == run_id]
                error("Num configurations found with id: {} is: {} during stat-testing!".format(run_id, len(conf)), len(conf) != 1)
                conf = conf[0]
                # get variables
                df_row = {k: v for (k, v) in conf.ddict.items() if k != "id"}
                for score in  results[run_id][run_mode][measure][label_aggregation]['folds']:
                    df_row["score"] = score
                    df_inputs.append(deepcopy(df_row))
        except:
            warning("Encountered invalid results accessors: {}".format((run_mode, measure, label_aggregation)))
            continue
        data = pd.DataFrame(df_inputs)
        inst = instantiator.Instantiator()
        stat_test = inst.create(method)
 
        for v, variable in enumerate(testable_variables):
            info("Experiment variable {}/{}: {}".format(v+1, len(testable_variables), variable))
            if limit_variables is not None:
                if variable not in limit_variables:
                    continue
            if len(data[variable]) == len(set(data[variable])):
                warning("Skipping testing for parameter [{}] due to having only 1 observation per value".format(variable))
                continue
            if len(set(data[variable])) == 1:
                warning("Skipping testing for parameter [{}] due to having only 1 unique parameter value: {}".format(variable, data[variable].values[0]))
                continue
            stat_result = stat_test.run(data["score"], data[variable])
            stat_test.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        help="Configuration .yml file for the run.")
    parser.add_argument("--only-report",
                        help="Do not run, just report results.",
                        action="store_true",
                        dest="only_report")
    parser.add_argument(
        "--force-dir",
        help=
        "Force the experiment directory to the relevant input argument (yml path or basedir)",
        action="store_true",
        dest="force_dir")

    parser.add_argument(
        "--no-config-check",
        help=
        "Do not examine that the input configuration matches the one in the experiment directory, if it exists.",
        action="store_true",
        dest="no_config_check")

    parser.add_argument(
        "--restart",
        help= "Restart the expeirments from scratch, deleting the output experiments directory.",
        action="store_true",
        dest="restart")

    args = parser.parse_args()
    main(args.config_file, args.only_report, args.force_dir, args.no_config_check, args.restart)
