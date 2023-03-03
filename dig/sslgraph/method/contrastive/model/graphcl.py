from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView, NodeTranslation
import random
import torch.nn as nn

class GraphCL(Contrastive):
    r"""    
    Contrastive learning method proposed in the paper `Graph Contrastive Learning with 
    Augmentations <https://arxiv.org/abs/2010.13902>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_graphcl.ipynb>`_ for
    an example of usage.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GraphCL`.
    
    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, args, **kwargs):  #method=None,
        
        views_fn = []
        self.z_dim = args.z_dim
        self.proj = args.proj
        self.aug_1, self.aug_2 = args.aug_1, args.aug_2
        self.tau = args.tau
        self.dropout = args.dropout_rate
        self.aug_ratio = args.aug_ratio
        self.device = args.device
        self.encoder = args.encoder
        
        for aug in [self.aug_1, self.aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'ETKDG1':
                views_fn.append(NodeTranslation(method='ETKDG1', device=self.device))
            elif aug == 'ETKDG2':
                views_fn.append(NodeTranslation(method='ETKDG2', device=self.device))
            elif aug == 'UFF1':
                views_fn.append(NodeTranslation(method='UFF1', device=self.device))
            elif aug == 'UFF2':
                views_fn.append(NodeTranslation(method='UFF2', device=self.device))
            elif aug == 'MMFF1':
                views_fn.append(NodeTranslation(method='MMFF1', device=self.device))
            elif aug == 'MMFF2':
                views_fn.append(NodeTranslation(method='MMFF2', device=self.device))
            elif aug == 'MMFF3':
                views_fn.append(NodeTranslation(method='MMFF3', device=self.device))
            elif aug == 'MMFF4':
                views_fn.append(NodeTranslation(method='MMFF4', device=self.device))
            elif aug == 'rotation':
                views_fn.append(NodeTranslation(method='rotation', device=self.device))
            elif aug == 'noise':
                views_fn.append(NodeTranslation(method='noise', device=self.device))
            elif aug == 'dropN':
                views_fn.append(UniformSample(encoder=self.encoder, ratio=self.aug_ratio, device=self.device))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(encoder=self.encoder, mask_ratio=self.aug_ratio))
            elif aug == 'random':
                method = random.choice(['MMFF1', 'MMFF4'])
                
                canditates = [NodeTranslation(method=method),
                              NodeTranslation(method='rotation'),
                              NodeTranslation(method='noise'),
                              UniformSample(ratio=self.aug_ratio)]
                views_fn.append(RandomView(canditates))
               
            elif aug == 'MMFFrandom':
                method = random.sample(['MMFF1', 'MMFF2', 'MMFF3', 'MMFF4'], 2)
                canditates = [NodeTranslation(method=method[0], device=self.device),
                              NodeTranslation(method=method[1], device=self.device)]
                views_fn.append(RandomView(canditates))
                
            elif aug == 'random3':
                method = random.choice(['MMFF1', 'MMFF4'])
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception("Aug must be from [MMFF1, MMFF2, 'rotation', 'noise', dropN', 'maskN', 'random2', 'random3', 'random4'] or None.")
        
        super(GraphCL, self).__init__(args, objective='NCE',
                                      views_fn=views_fn,
                                      z_dim=self.z_dim,
                                      proj=self.proj,
                                      dropout = self.dropout,
                                      **kwargs)
        
    def train(self, encoders, data_loader, optimizer, scheduler, epochs, per_epoch_out=True):
        # GraphCL removes projection heads after pre-training
        encoders = encoders.to(self.device)
        for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                    optimizer, scheduler, epochs, per_epoch_out):
            yield enc
