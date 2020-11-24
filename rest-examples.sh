#!/usr/bin/env bash

if [[ -z  $( which curl ) ]]; then 
	echo "Need curl to run the rest examples"
	exit 1
fi

# settings
#########
echo "asdf"
# endpoint
ip="http://localhost"
port=9998

# input data (array of strings)
data='[ "Καλημέρα τι κάνετε", "Τι ώρα είναι κύριε ηρεμήστε παρακαλώ"]'

# input config as dictionary. Parameters:
# binary_treshold: Float. lowermost confidence threshold to advance a word to multiclass classif.
# multiclass_treshold: Float. lowermost confidence threshold to consider a replacement word
# top_k: Integer. How many prediction scores / classes to retrieve per token.
params='{ "binary_threshold": 0.95, "multiclass_threshold": 0.5, "top_k": 5}'

#########


baseurl="${ip}:${port}/test"
dat="{ \"text\": $data, \"params\": $params}"

# do the query
curl -X POST "${baseurl}" -d "${dat}" -H 'Content-Type: application/json'

# returns a dictionary:
# {
#     "multistageclassif_report": {
#         "results": [......],
#         "input_params": {
#             "binary_threshold": 0.95,
#             "multiclass_threshold": 0.5,
#             "top_k": 3
#         },
#         "messages": []
# }

# where:
# "multistageclassif_report"  # the name of the results data bundle
# "results": [......] # array with the results themselves (described below).
# "input_params": { ... } # dictionary replicating any input parameters fed to the request that produced the result
# "messages": []  # List of strings with messages. Current possible messages:
#                   Inform a default topk (=5) if none was supplied.

# results: List of [res1, res2, ..., resN], where res_i is a dictionary with the multistep classification results of the i-th input instance (input text to be classified).
# An example res_i is below:

#  {
#         "instance": 0,
#         "data": [
#           "καλημέρα",
#           "τι",
#           "κάνετε"
#         ],
#         "predictions": [
#           {
#             "step": 0,
#             "survivors": [
#               1
#             ],
#             "topk_preds": [
#               [
#                 0.8828849077706096,
#                 0.11711509222939043
#               ],
#               [
#                 0.9736009724567856,
#                 0.026399027543214437
#               ],
#               [
#                 0.9323038420240538,
#                 0.06769615797594619
#               ]
#             ],
#             "topk_classes": [
#               [
#                 "replace",
#                 "keep"
#               ],
#               [
#                 "replace",
#                 "keep"
#               ],
#               [
#                 "replace",
#                 "keep"
#               ]
#             ]
#           },
#           {
#             "step": 1,
#             "survivors": [
#               0
#             ],
#             "topk_preds": [
#               [
#                 0.4119197410608065,
#                 0.1747497928022213,
#                 0.09277467907717561
#               ]
#             ],
#             "topk_classes": [
#               [
#                 "καλε",
#                 "ουσιωδεις",
#                 "αφαιρουνται"
#               ]
#             ]
#           }
#         ],
#         "total_survivors": [
#           1
#         ],
#         "total_topk_preds": [
#           [
#             0.4119197410608065,
#             0.1747497928022213,
#             0.09277467907717561
#           ]
#         ],
#         "total_topk_classes": [
#           [
#             "καλε",
#             "ουσιωδεις",
#             "αφαιρουνται"
#           ]
#         ]
#       }


# Keys & values of each res_i (TL;DR further below):

# "instance": Contains the the zero-addressed index of the instance.
#             E.g. 0 for the first text instance fed to the api.
# "data": List [tok1, tok2, ..., tokT] of individual word tokens of the corresponding
#         input instance, e.g. ["καλημέρα", "τι", "κάνετε"]. NOTE: if any data
#         preprocessing is performed (e.g. stopword filtering, stemming, etc.)
#         this will differ from the input instance. By default no preprocessing's done.
# "predictions": List of [p1, p2, ... pM], where p_l the l-th classification in the
#                M-long multistep classification process. E.g. in a binary & multiclass
#                two-step process there will be two elements in the array.

# Keys & values of each multistep prediction result p_i:
# "step": Contains an integer with the classification step number. Starts at 0.
# "topk_preds": List of [tp1, tp2, ..., tpT], where tp_i contains the top K predictions
#               for the i-th instance token in descending order. K can be set as an
#               input parameter in the api. K will be <= the total available number
#               of classes.
# "topk_classes": List of [tc1, tc2, ..., tcT], where tc_i contains the classes
#                 corresponding to the top K predictions in the "topk_preds" field,
#                 for the i-th instance token.
# "survivors": Integer list, with each element i 0<=i<T-1 corresponds to the T instance
#              tokens. If i is in the list, then the i-th token survived the
#              thresholding operation of this classif. step and is fed to the next.
#              NOTE: Each classif. step is performed only for data surviving the
#              threshold, i.e. inputs & results for classif. step m > 0 only contain
#              survivors of step m-1.

# (TL;DR): To obtain results for input instance tokens that survive the entire classification chain, you can use the fields:
# "final_survivors": Like "survivors", but for all classification steps: i.e, a list of zero-addressed indices of instance tokens that survive tresholding for *all* classif. steps.
# "final_topk_preds"] = Similar to "topk_preds", but only for the final survivors tokens and corresponding to the final classification step.
# "final_topk_classes"] = Similar to "topk_classes", but only for the final survivors tokens and corresponding to the final classification step.
