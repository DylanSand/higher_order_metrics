
######################################################################################## LIBRARY

import torch
import torch.nn.functional as F
import pickle
from torch_geometric.nn import GCNConv, GATConv, GINConv
from CONFIG import NUM_TESTS_PER_GRAPH, MINIMUM_START, EARLY_STOP_LIMIT NUM_EPOCHS


######################################################################################## MAIN CODE

# Defines our model class
class Net(torch.nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, is_fc, gnn_type):
        super(Net, self).__init__()
        self.is_fc = is_fc
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(NUM_FEATURES, 16)
            self.conv2 = GCNConv(16, NUM_CLASSES)
        if gnn_type == 'GIN':
            self.lin1 = torch.nn.Linear(NUM_FEATURES, 16)
            self.lin2 = torch.nn.Linear(16, NUM_CLASSES)
            self.conv1 = GINConv(self.lin1)
            self.conv2 = GINConv(self.lin2)
        if gnn_type == 'GAT':
            self.conv1 = GATConv(NUM_FEATURES, 16)
            self.conv2 = GATConv(16, NUM_CLASSES)
        if is_fc:
            self.fc_layer = torch.nn.Linear(16, NUM_CLASSES)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        if self.is_fc:
            x = self.fc_layer(x)
        else:
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# This is the function that runs our model experiment
def runGNNModel(real_data, NUM_FEATURES, NUM_CLASSES, is_fc, gnn_type):
    if is_fc:
        print('Running FC model ' + str(NUM_TESTS_PER_GRAPH) + ' times...')
    else:
        print('Running normal model ' + str(NUM_TESTS_PER_GRAPH) + ' times...')
    graph_accs = []
    for x in range(NUM_TESTS_PER_GRAPH):

        # Set up learning
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(NUM_FEATURES, NUM_CLASSES, is_fc, gnn_type).to(device)
        data = real_data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Do learning
        model.train()
        best_val_acc = 0.0
        training_count = 0
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch < MINIMUM_START:
                #print('In burnout...')
                f = open("current_model.p", "wb")
                pickle.dump(model, f)
                f.close()
                model.eval()
                _, pred = model(data).max(dim=1)
                correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                best_val_acc = correct / int(data.val_mask.sum())
            else:
                model.eval()
                _, pred = model(data).max(dim=1)
                correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                cur_val_acc = correct / int(data.val_mask.sum())
                if cur_val_acc < best_val_acc:
                    training_count = training_count + 1
                    #print('Current acc is worse, training count now ' + str(training_count))
                    if training_count >= EARLY_STOP_LIMIT:
                        f = open("current_model.p", "rb")
                        model = pickle.load(f)
                        f.close()
                        #print('Breaking...')
                        break
                else:
                    #print('Current acc is better...')
                    training_count = 0
                    best_val_acc = cur_val_acc
                    f = open("current_model.p", "wb")
                    pickle.dump(model, f)
                    f.close()

        # Analyze results
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        graph_accs.append(acc)
    cur_graph_acc = sum(graph_accs) / float(NUM_TESTS_PER_GRAPH)
    return cur_graph_acc
