
######################################################################################## LIBRARY
import torch
from torch_geometric.utils import degree, is_undirected, to_undirected, dense_to_sparse, to_networkx
from torch_geometric.datasets import Planetoid, Coauthor, Actor, Amazon, SNAPDataset, GNNBenchmarkDataset
import numpy as np

import sys
sys.path.insert(1, './MixHop-and-N-GCN')
sys.path.insert(1, './MixHop-and-N-GCN/')
sys.path.insert(1, './MixHop-and-N-GCN/src')
sys.path.insert(1, './MixHop-and-N-GCN/src/')
from metricCalc import doCycleProbMetricCalc
from GNNModel import runGNNModel
from param_parser import parameter_parser
from trainer_and_networks import Trainer
from CONFIG import NUM_CLASSES, TRAIN_PERCENT, VAL_PERCENT

np.random.seed(np.random.randint((2**32) - 4))


# Arguments:
# 
# dataset_id: String = << The name of the dataset we want to experiment on. >>
# DO_METRIC: boolean = << Whether or not to additionally do analysis using the "clyclic probability" metric. >>
# input_path: String = << The path to the "graphs" folder that contains the raw dataset files from Pytorch Geometric. >>
# 
# Returns:
# 
# A list of all of the experimental results
# 
def runExperimentMixHop(dataset_id, DO_METRIC, input_path):
    return_list = []
    ######################################################################################## GRAPH CREATION
    # Set up real data
    print("Now running experiment on: " + dataset_id + "\nWith GNN type: MixHop")

    return_list.append(dataset_id + '_MixHop')

    if dataset_id == 'Cora':
        dataset_name = 'Cora'
        over_data = Planetoid(input_path + '/graphs/' + dataset_name, dataset_name, 'public')
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'CiteSeer':
        dataset_name = 'CiteSeer'
        over_data = Planetoid(input_path + '/graphs/' + dataset_name, dataset_name, 'public')
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'PubMed':
        dataset_name = 'PubMed'
        over_data = Planetoid(input_path + '/graphs/' + dataset_name, dataset_name, 'public')
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'Actor':
        dataset_name = 'Actor'
        over_data = Actor(input_path + '/graphs/' + dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'Photo':
        dataset_name = 'Photo'
        over_data = Amazon(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'Computers':
        dataset_name = 'Computers'
        over_data = Amazon(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'CS':
        dataset_name = 'CS'
        over_data = Coauthor(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'Physics':
        dataset_name = 'Physics'
        over_data = Coauthor(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'ego-facebook':
        dataset_name = 'ego-facebook'
        over_data = SNAPDataset(input_path + '/graphs/' + dataset_name, dataset_name)
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = 24
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'ego-gplus':
        dataset_name = 'ego-gplus'
        over_data = SNAPDataset(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'ego-twitter':
        dataset_name = 'ego-twitter'
        over_data = SNAPDataset(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_CLASSES = over_data.num_classes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'wiki-vote':
        dataset_name = 'wiki-vote'
        over_data = SNAPDataset(input_path + '/graphs/' + dataset_name, dataset_name)
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_FEATURES = over_data.num_node_features

    if dataset_id == 'PATTERN':
        dataset_name = 'PATTERN'
        over_data = GNNBenchmarkDataset(input_path + '/graphs/' + dataset_name, dataset_name, split='test')
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_FEATURES = over_data.num_node_features
        NUM_CLASSES = 2

    if dataset_id == 'CLUSTER':
        dataset_name = 'CLUSTER'
        over_data = GNNBenchmarkDataset(input_path + '/graphs/' + dataset_name, dataset_name, split='test')
        real_data = over_data[0]
        NUM_NODES = real_data.num_nodes
        NUM_FEATURES = over_data.num_node_features
        NUM_CLASSES = 6


    if not is_undirected(real_data.edge_index, num_nodes=NUM_NODES):
        real_data.edge_index = to_undirected(real_data.edge_index, num_nodes=NUM_NODES)

    # Make train/test masks
    TEST_PERCENT = 1.0 - (TRAIN_PERCENT + VAL_PERCENT)
    num_train = int(float(NUM_NODES) * TRAIN_PERCENT)
    num_val = int(float(NUM_NODES) * VAL_PERCENT)
    train_mask = torch.tensor(np.array([True] * NUM_NODES))
    val_mask = torch.tensor(np.array([False] * NUM_NODES))
    test_mask = torch.tensor(np.array([False] * NUM_NODES))
    chosen_not_train = np.random.choice(NUM_NODES, NUM_NODES - num_train, replace=False)
    for cur_index in chosen_not_train:
        train_mask[cur_index] = False
        test_mask[cur_index] = True
    chosen_val = np.random.choice(chosen_not_train, num_val, replace=False)
    for cur_index in chosen_val:
        val_mask[cur_index] = True
        test_mask[cur_index] = False

    # Augment Data object
    real_data.train_mask = train_mask
    real_data.test_mask = test_mask
    real_data.val_mask = val_mask
    real_data.degree = degree(index=real_data.edge_index[0], num_nodes=real_data.num_nodes)

    ######################################################################################## METRIC CALC
    return_list.append('CycleProbMetric')
    if DO_METRIC:
        cur_metric = doCycleProbMetricCalc(real_data)
        print('Cycle probability metric has been calculated: ' + str(cur_metric))
        return_list.append(cur_metric)
    else:
        print('Did not run cycle probability metric')
        return_list.append(0.0)

    ######################################################################################## GNN LEARNING
    return_list.append('MixHopGNNAcc')

    graph_mix = to_networkx(real_data)
    features_mix = dict()
    features_mix["dimensions"] = real_data.x.shape
    index, values = dense_to_sparse(real_data.x)
    features_mix["indices"] = torch.LongTensor(index)
    features_mix["values"] = torch.FloatTensor(values)
    target_mix = torch.LongTensor(real_data.y)

    args = parameter_parser()
    trainer = Trainer(args, graph_mix, features_mix, target_mix, True)
    trainer.fit()
    trainer.evaluate_architecture()
    args = trainer.reset_architecture()
    trainer = Trainer(args, graph_mix, features_mix, target_mix, False)
    trainer.fit()

    cur_graph_acc = 0.0
    print('Ran normal MixHop: ' + str(cur_graph_acc))
    return_list.append(cur_graph_acc)

    print('Final results: ' + str(return_list))
    print('')
    return return_list

