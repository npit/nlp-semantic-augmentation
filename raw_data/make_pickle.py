import pickle
import os
import csv
import pandas
import sys

# Script to picklify raw data

if len(sys.argv) < 1:
    print("input_path <resource_type>")
    exit(1)

input_path = sys.argv[1]
input_resource = sys.argv[2]
print("Pickling {}".format(input_path))
if input_resource  == "glove":
    data = pandas.read_csv(input_path, index_col = 0, header=None, sep=" ", quoting=csv.QUOTE_NONE)
    dim = data.shape[-1]
elif input_resource  == "fasttext":
    import pdb;pdb.set_trace()
    data = pandas.read_csv(input_path, index_col = 0, header=None, skiprows=1, sep=" ", quoting=csv.QUOTE_NONE)
    dim = data.shape[-1]
else:
    print("Don't know how to pickle that.")
    exit(1)

outname = "{}_dim{}".format(input_resource, dim)
outpath = os.path.join(os.path.dirname(input_path), outname + ".pickle")
print("Writing to {}".format(outpath))
with open(outpath, "wb") as f:
    pickle.dump(data, f)
