from os.path import join, exists, splitext, basename, dirname
from os import listdir, makedirs
import pickle
import nltk
from utils import tictoc, error, info, debug, warning, write_pickled, read_pickled, shapes_list, nltk_download
import numpy as np
from semantic.semantic_resource import  SemanticResource
import json
import urllib
from scipy import spatial
import spotlight
import yaml
import requests

import defs




