from os.path import join, exists, isabs, basename, splitext
from os import makedirs
import subprocess
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
from utils import info, error, datetime_str, aslist

"""Script to produce large-scale semantic neural augmentation experiments

The experiment variation parameters should be in a "params" field in the configuration file.
Values in a list are interpreted as different parameters (so for list literal values, add an additional list wrap)
"""

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
    if type(ddict[key]) == dict:
        prev_keys.append(key)
        res = traverse_dict(ddict[key], None, prev_keys)
    else:
        val = ddict[key]
        if type(val) != list:
            val = [val]
        res = (val, prev_keys + [key])
    return res


def make_configs(base_config, run_dir, sources_dir="./"):
    vars = []
    params = base_config["params"]
    base_raw_folder = base_config["folders"]["raw_data"]
    base_serialization_folder = base_config["folders"]["serialization"]
    for val in sorted(params.keys()):
        seqs = traverse_dict(params, val, [])
        vars.extend(seqs)
    configs, run_ids = [], []
    vars = sorted(vars, key=lambda x: str(x)[2])

    values = [v[0] for v in vars]
    names = [v[1] for v in vars]

    for combo in itertools.product(*values):
        conf = deepcopy(base_config)
        name_components = []
        for v, value in enumerate(combo):
            lconf = conf
            # make sure it's directory-friendly
            if type(value) == list:
                val = "-".join(list(map(str, value)))
                name_components.append(val)
            else:
                name_components.append(basename(splitext(str(value))[0]))
            key_chain = names[v]
            for key in key_chain[:-1]:
                if key not in lconf:
                    lconf[key] = {}
                lconf = lconf[key]
            lconf[key_chain[-1]] = value
        # dirs
        run_id = "_".join(name_components)
        print("Built run id:", run_id)
        conf["folders"]["run"] = join(run_dir, run_id)
        if isabs(base_serialization_folder):
            conf["folders"]["serialization"] = base_serialization_folder
        else:
            conf["folders"]["serialization"] = join(sources_dir, base_serialization_folder)
        if isabs(base_raw_folder):
            conf["folders"]["raw_data"] = base_raw_folder
        else:
            conf["folders"]["raw_data"] = join(sources_dir, base_raw_folder)
        configs.append(conf)
        run_ids.append(run_id)
    return configs, run_ids


# make a run id name out of a list of nested dict keys and a configuration dict
def make_run_ids(keychains, confs):
    names = []
    for conf in confs:
        name_components = []
        for keychain in keychains:
            name_components.append(reduce(dict.get, keychain, conf))
        names.append("_".join(map(str, name_components)))
    return names


def main(config_file="large.config.yml"):
    # settable parameters
    ############################################################

    # config file
    email = "pittarasnikif@gmail.com"
    passw = None

    ############################################################

    # set the expeirment parameters via a configuration list
    conf = yaml.load(open(config_file))
    # evaluation measures
    exps = conf["experiments"]
    eval_measures = aslist(exps["measures"]) if "measures" in exps else ["f1-score", "accuracy"]
    aggr_measures = aslist(exps["aggregation"]) if "aggregation" in exps else ["macro", "micro"]
    stat_functions = aslist(exps["stats"]) if "stats" in exps else ["mean"]
    run_types = aslist(exps["run_types"]) if "run_types" in exps else ["run"]

    # folder to run experiments in
    run_dir = exps["run_folder"]
    # folder where run scripts are
    sources_dir = exps["sources_dir"] if "sources_dir" in exps else "./"

    configs, run_ids = make_configs(conf, run_dir, sources_dir)

    # virtualenv folder
    venv_dir = conf["experiments"]["venv"] if "venv" in conf["experiments"] else None
    # results csv file
    # results_file = conf["experiments"]["results_file"]
    results_file = join(run_dir, "run_results.csv")

    # mail
    do_send_mail = exps["send_mail"] if "send_mail" in exps else None
    if do_send_mail:
        passw = getpass.getpass()

    # dir checks
    if venv_dir and not exists(venv_dir):
        error("Virtualenv dir {} not found".format(venv_dir))
    if not exists(run_dir):
        info("Run dir {} not found, creating.".format(run_dir))
        makedirs(run_dir)

    # copy the configuration file in the target directory
    copied_conf = join(run_dir, basename(config_file))
    if exists(copied_conf):
        # make sure it's the same effing config
        cconf = yaml.load(copied_conf)
        if cconf != conf:
            info("The original config differs from the one in the experiment directory!")
    else:
        info("Copying experiments configuration at {}".format(copied_conf))
        with open(copied_conf, "w") as f:
            yaml.dump(conf, f)

    # logging
    level = logging._nameToLevel[conf["log_level"].upper()]
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    logfile = os.path.join(run_dir, "experiments_{}.log".format(datetime_str()))

    chandler = logging.StreamHandler()
    chandler.setFormatter(formatter)
    chandler.setLevel(level)
    logging.getLogger().addHandler(chandler)

    fhandler = logging.FileHandler(logfile)
    fhandler.setLevel(level)
    fhandler.setFormatter(formatter)
    logging.getLogger().addHandler(fhandler)
    logging.getLogger().setLevel(logging.DEBUG)

    results = {}

    #################################################################################

    # prelim experiments
    for conf_index, (conf, run_id) in enumerate(zip(configs, run_ids)):
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
            if exists(error_file):
                os.remove(error_file)
            makedirs(experiment_dir, exist_ok=True)

            conf_path = join(experiment_dir, "config.yml")
            with open(conf_path, "w") as f:
                yaml.dump(conf, f)
            # write the run script file
            script_path = join(experiment_dir, "run.sh")
            with open(script_path, "w") as f:
                if venv_dir:
                    f.write("source \"{}/bin/activate\"".format(venv_dir))
                f.write("cd \"{}\"\n".format(sources_dir))
                f.write("python3 \"{}\" --config_file \"{}\" && touch \"{}\" && exit 0\n".format(join(sources_dir, "main.py"), conf_path, completed_file))
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
            res_data = pickle.load(f)
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
        df = pd.DataFrame.from_dict(print_vals, orient='index')
        print(df.to_string())
        total_results[stat] = print_vals
    info("Writing these results to file {}".format(results_file))
    total_df = pd.DataFrame.from_dict(total_results, orient='index')
    total_df.to_csv(results_file)

    # [info(msg) for msg in messages]
    if do_send_mail:
        sendmail(email, passw, "run complete.")


if __name__ == "__main__":
    main()
