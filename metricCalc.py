
######################################################################################## LIBRARY

import torch
from torch_sparse import spspmm
from CONFIG import METRIC_RANDOM_WALK_LENGTH


######################################################################################## MAIN CODE

def getCycleMat(ind1, val1, size1, ind2, val2, size2, cycle_num):
    cur_ind = ind1
    cur_val = val1
    for x in range(cycle_num - 1):
        cur_ind, cur_val = spspmm(cur_ind, cur_val, ind1, val1, size1, size1, size1)
    # DO FINAL MM OR JUST TAKE EDGE TRANITION TO CORRECT POWER?
    return cur_ind, cur_val
    #return spspmm(cur_ind, cur_val, ind2, val2, size1, size1, size2)

def doCycleProbMetricCalc(real_data):

    # Set up metric opbjects
    edge_transition_index = [[], []]
    edge_transition_value = []
    edge_adjacency_index = [[], []]
    edge_adjacency_value = []

    # Calculate "edge transition" matrix
    print('Making edge transition matrix:')
    for x in range(real_data.edge_index.shape[1]):
        print('ET: ' + str((x / real_data.edge_index.shape[1]) * 100) + '%')
        for y in range(real_data.edge_index.shape[1]):
            if not x == y:
                v_1 = int(real_data.edge_index[1][x])
                u_2 = int(real_data.edge_index[0][y])
                if v_1 == u_2:
                    v_2 = int(real_data.edge_index[1][y])
                    u_1 = int(real_data.edge_index[0][x])
                    if not v_2 == u_1:
                        out_num = 1.0 / (float(real_data.degree[v_1]) - 1.0)
                        edge_transition_index[0].append(x)
                        edge_transition_index[1].append(y)
                        edge_transition_value.append(out_num)
    edge_transition_index = torch.tensor(edge_transition_index, dtype=torch.long)
    edge_transition_value = torch.tensor(edge_transition_value, dtype=torch.double)

    # Calculate "edge adjacency" matrix
    print('Making edge adjacency matrix:')
    for x in range(real_data.edge_index.shape[1]):
        print('EA: ' + str((x / real_data.edge_index.shape[1]) * 100) + '%')
        for y in range(real_data.edge_index.shape[1]):
            out_num = 0
            if not x == y:
                v_1 = int(real_data.edge_index[1][x])
                u_2 = int(real_data.edge_index[0][y])
                if v_1 == u_2:
                    out_num = 1
            if out_num != 0:
                edge_adjacency_index[0].append(x)
                edge_adjacency_index[1].append(y)
                edge_adjacency_value.append(out_num)
    edge_adjacency_index = torch.tensor(edge_adjacency_index, dtype=torch.long)
    edge_adjacency_value = torch.tensor(edge_adjacency_value, dtype=torch.double)

    # Do W-Out calculations from stable distribution on edges
    #t = METRIC_RANDOM_WALK_LENGTH
    #alpha = 0.15
    #cycle_result_index, cycle_result_value = getCycleMat(edge_transition_index, edge_transition_value, real_data.edge_index.shape[1], edge_adjacency_index, edge_adjacency_value, real_data.edge_index.shape[1], t)
    #bold_one = []
    #for i in range(real_data.edge_index.shape[1]):
    #    bold_one.append([1.0])
    #bold_one = torch.tensor(bold_one, dtype=torch.double)
    #w_out = spmm(cycle_result_index, cycle_result_value, real_data.edge_index.shape[1], real_data.edge_index.shape[1], bold_one)
    #scalar = 0.0
    #for i in range(t):
    #    scalar = scalar + (1.0 - alpha)**(i + 1)
    #scalar = scalar * (alpha / real_data.edge_index.shape[1])
    #w_out = scalar * w_out

    # Do average cycle probability metric calculation
    master_node_probs = [1.0] * real_data.edge_index.shape[1]
    master_node_probs = torch.tensor(master_node_probs)
    ones_matrix = [1.0] * real_data.edge_index.shape[1]
    ones_matrix = torch.tensor(ones_matrix)
    for i in range(METRIC_RANDOM_WALK_LENGTH):
        if i+1 < 3:
            continue
        cycle_result_index, cycle_result_value = getCycleMat(edge_transition_index, edge_transition_value, real_data.edge_index.shape[1], edge_adjacency_index, edge_adjacency_value, real_data.edge_index.shape[1], i+1)
        # Calculate average probability of there being a cycle
        node_probs = [0.0] * real_data.edge_index.shape[1]
        tmp_ind_list = cycle_result_index[1].tolist()
        tmp_val_list = cycle_result_value.tolist()
        for i, x in enumerate(cycle_result_index[0].tolist()):
            if tmp_ind_list[i] == x:
                node_probs[x] = tmp_val_list[i]
        node_probs = torch.tensor(node_probs)
        node_probs = ones_matrix - node_probs
        master_node_probs = master_node_probs * node_probs
    master_node_probs = ones_matrix - master_node_probs
    master_node_probs = master_node_probs.tolist()
    cur_metric = sum(master_node_probs) / float(len(master_node_probs))
    return cur_metric
