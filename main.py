
######################################################################################## LIBRARY
import getopt
import sys
from experimentRunner import runExperiment
from experimentRunnerMixHop import runExperimentMixHop
from CONFIG import datasets

input_path = ''
job_id = ''
opts, args = getopt.getopt(sys.argv[1:],"d:",["input_path="])
for opt, arg in opts:
    if opt in ("-d", "--input_path"):
        job_id = arg[11:-2]
        input_path = arg + ''
    else:
        sys.exit()


######################################################################################## MAIN CODE

final_data = []

for dataset in datasets:
    final_data.append(runExperiment(dataset, False, input_path, 'GCN'))
    final_data.append(runExperiment(dataset, False, input_path, 'GAT'))
    final_data.append(runExperiment(dataset, False, input_path, 'GIN'))
    final_data.append(runExperimentMixHop(dataset, False, input_path))

print(final_data)
