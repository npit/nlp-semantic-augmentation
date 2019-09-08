import smtplib
from collections import OrderedDict
from os.path import join

import yaml

from experiments.variable_config import VariableConf
from utils import error, info


def sendmail(mail, passw, msg, title="nle"):
    """Mail sender function"""
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

    BODY = '\r\n'.join([
        'To: %s' % TO,
        'From: %s' % gmail_sender,
        'Subject: %s' % SUBJECT, '', TEXT
    ])
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
    """Examine if a sequence of nested keys exists in a dict"""
    for key in key_seq:
        try:
            ddict = ddict[key]
        except KeyError:
            return False
    return True

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
                info("Omitting incompatible config {} with bad entries: {}".
                     format(conf.id, bad_combo))
                break
        if not bad_conf:
            out_conf.append(conf)
    return out_conf


# get a nested dict value from a list of keys
def get_kseq_value(kseq, ddict):
    res = ddict
    for k in kseq:
        res = res[k]
    return res

def compare_dicts(dict1, dict2):
    """Compares two dictionaries reporting differences"""
    for k, v in dict1.items():
        if k not in dict2:
            return False, "The key [{}] in first dict: {} missing from the second dict: {}".format(
                k, dict1, dict2)
        if type(v) in [dict, OrderedDict]:
            eq, diff = compare_dicts(dict1[k], dict2[k])
            if not eq:
                return eq, diff
        if v != dict2[k]:
            return False, "Differing values: {} and {}, in dicts {}, {} with key {}".format(
                v, dict2[k], dict1, dict2, k)
        else:
            pass
            # print("same values: {} for dict parts {}".format(v, dict1))
    return True, None
