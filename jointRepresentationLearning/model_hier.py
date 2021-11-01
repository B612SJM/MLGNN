import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Tuple, List, Dict
from jointRepresentationLearning.encoder import GCNEncoder, GCNEncoderWithFeatures
from jointRepresentationLearning.decoder import InnerProductDecoder
from features import BatchMolGraph
from models.mpn import MPN
from data.mol_tree import Vocab
from argparse import Namespace
import numpy as np
from sklearn.svm import SVC, SVR
import joblib

from models.mpn import PairMPN
from models.ggnn import GGNN
from models.smiles import SmilesNN
from models.jtnn_enc import JTNNEncoder
from models.jt_mpn import JunctionTreeGraphNN
from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights
from models.feature_loader import Mol2vecLoader
from models.pooling import *
from data.data import mol2sentence
from .model_info import shortcut
from .deepinfomax import GcnInfomax

class HierGlobalGCN(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(HierGlobalGCN, self).__init__()
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.bias = bias
        self.sparse = sparse
        self.args = args
        self.create_encoder(args)       
        self.global_enc = self.select_encoder(args)
        self.dec_local = InnerProductDecoder(args.hidden_size)
        self.dec_global = InnerProductDecoder(args.hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.DGI_setup()
        self.create_ffn(args)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        if not args.smiles_based:
            if args.graph_encoder == 'ggnn':
                self.encoder = GGNN(args)
            else:
                self.encoder = MPN(args)
        else:
            self.encoder = SmilesNN(args)

        if args.jt:
            self.encoder = JTNNEncoder(vocab, args.hidden_size) if args.jt_encoder == 'tree' else \
                JunctionTreeGraphNN(args)

        return self.encoder

    def select_encoder(self, args: Namespace):
        return GCNEncoderWithFeatures(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)

    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        self.fusion_ffn_local = nn.Linear(args.hidden_size+args.input_features_size, args.ffn_hidden_size)#  (300+vector_size*length) * 264  args.length
        self.fusion_ffn_global = nn.Linear(args.gcn_hidden3, args.ffn_hidden_size)#264*264
        ffn = []
        # after fusion layer
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),#264*264
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),#264*drug_nums
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout

    def DGI_setup(self):
        self.DGI_model = GcnInfomax(self.args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        ------------------------------------------------------
        使用额外特征时使用下面
        ------------------------------------------------------
        """
        if self.args.use_input_features:
            smiles_batch = batch[0]
            features_batch = batch[1]
        else:
            smiles_batch=batch
            features_batch=None
        # print(self.encoder)

        """
        feat_orig:  [544,300]
        feat: [544,300]
        fused_feat: [544,264]
        output: [544,544]
        outputs: [544,544]
        outputs_l: [295936]
        """
        feat_orig = self.encoder(smiles_batch, features_batch)

        feat = self.dropout(feat_orig)
        # print(feat.shape)

        fused_feat = self.fusion_ffn_local(feat)# args.hidden_size+args.input_features_size  **  264
        # print(fused_feat.shape)
        output = self.ffn(fused_feat)#264  ** 264;  264  **  drug_nums
        outputs = self.sigmoid(output)
        outputs_l = outputs.view(-1)
        # print("outpus_l.shape:"+outputs_l.shape)
        """
        embeddings: [544,264]
        feat_g: [544,264]
        fused_feat_g: [544,264]
        output_g: [544,544]
        outputs_: [544,544]
        outputs_g: [295936]
        """
        embeddings = self.global_enc(feat_orig, adj)


        feat_g = self.dropout(embeddings)#embedding维度是264
        fused_feat_g = self.fusion_ffn_global(feat_g)# fusion_ffn_global:264  ** 264
        output_g = self.ffn(fused_feat_g)#264*264  264*drug_nums
        outputs_ = self.sigmoid(output_g)
        outputs_g = outputs_.view(-1)
        local_embed = feat_orig


        if return_embeddings:
            return outputs_, embeddings
        return outputs_g, feat_orig, embeddings, outputs_l, outputs_g

    """
    outputs_g是经过全局GCN和两层的全连接网络后的预测结果 --preds
    feat_orig是原始的分子图特征 --feat
    embeddings是经过全局GCN后的分子embedding  --embed
    outputs_l是经过MPN后直接通过两层全连接网络后的预测结果 --re_feat
    outputs_g是preds --re_embed
    DGI_LOSS是
    """