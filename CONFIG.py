

######################################################################################## main.py

datasets = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'Actor', 'PATTERN', 'CLUSTER', 'CS', 'Physics']


######################################################################################## GNNModel.py

# Determines how many times a model is created and run
NUM_TESTS_PER_GRAPH = 20

# Determines how many epochs are garunteed to run before early stopping kicks in
MINIMUM_START = 20

# Determines how many epochs need to pass without a validation improvement before the model stops
EARLY_STOP_LIMIT = 8

# Determines the maximum number of epochs
NUM_EPOCHS = 200


######################################################################################## metricCalc.py

# Determines the maximum path length that the algorthim will use when searching for cycles
METRIC_RANDOM_WALK_LENGTH = 5


######################################################################################## experimentRunner.py + experimentRunnerMixHop.py

# How many classes should be used when the data does not specify it?
NUM_CLASSES = 4

# What percent of the data should be used as training data?
TRAIN_PERCENT = 0.4

# What percent of the data should be used as validation data?
VAL_PERCENT = 0.1
