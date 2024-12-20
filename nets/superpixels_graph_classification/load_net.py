"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.superpixels_graph_classification.gated_gcn_net import GatedGCNNet
from nets.superpixels_graph_classification.gcn_net import GCNNet, GCNMasker
from nets.superpixels_graph_classification.gat_net import GATNet
from nets.superpixels_graph_classification.graphsage_net import GraphSageNet
from nets.superpixels_graph_classification.gin_net import GINNet
from nets.superpixels_graph_classification.mo_net import MoNet as MoNet_
from nets.superpixels_graph_classification.mlp_net import MLPNet
from nets.superpixels_graph_classification.ring_gnn_net import RingGNNNet
from nets.superpixels_graph_classification.three_wl_gnn_net import ThreeWLGNNNet
from nets.superpixels_graph_classification.gcnii_net import GCNIINet
from nets.superpixels_graph_classification.GPro_Net import GProNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)
    
def GCNII(net_params):
    return GCNIINet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)

def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)

def Masker(net_params):
    return GCNMasker(net_params)

def GPro(net_params):
    return GProNet(net_params)



def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN,
        'GCNII':GCNII,
        'GPro': GPro
    }
        
    return models[MODEL_NAME](net_params)


def mask_model(net_params):

    return Masker(net_params)
