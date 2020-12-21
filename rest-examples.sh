#!/usr/bin/env bash

# Example script for running RESTful queries

if [[ -z  $( which curl ) ]]; then 
	echo "Need curl to run the rest examples"
	exit 1
fi

BIN_THRESH=0.9
MC_THRESH=0.5
DO_DEBUG=0
[ $# -gt 0 ] && BIN_THRESH="$1"
[ $# -gt 1 ] && MC_THRESH="$2"
[ $# -gt 2 ] && DO_DEBUG="$3"

# settings
#########
# endpoint
ip="http://localhost"
port=9998

# input data (array of strings)

data='["το 50% της αμοιβής στο νέο διορισθέντα διαιτητή Χαράλαμπο Μεϊδάνη, το παρόν δικαστήριο ομόφωνα αποφαίνεται ότι πρέπει", "θα φωνάξει τον κατασκευαστή που είανι στην προσωρινή παραλαβή","παραδοθηκε με ενα το βαλαμε μες τον τεχνικο φακελο της προσωρινης παραλαβης τους το ειπαμε και προφορικα ειχε ληξει αυτο το θεμα", "ολα αυτα εγιναν μετα απο την κληση πολυ μετα απο την κληση για οριστικη παραλαβη και οπωσδηποτε οπως εμεις ισχυριζομαστε και μετα απο την οριστικη παραλαβη"]'
# input config as dictionary. Parameters:
# binary_treshold: Float. lowermost confidence threshold to advance a word to multiclass classif.
# multiclass_treshold: Float. lowermost confidence threshold to consider a replacement word
# top_k: Integer. How many prediction scores / classes to retrieve per token.
params="{ \"binary_threshold\": $BIN_THRESH , \"multiclass_threshold\": $MC_THRESH , \"debug\": $DO_DEBUG, \"top_k\": 5}"
# params="{ \"binary_threshold\": $BIN_THRESH , \"multiclass_threshold\": $MC_THRESH , \"debug\": $DO_DEBUG, \"top_k\": 5, \"skip_step_predictions\": 1}"

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
