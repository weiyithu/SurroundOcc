#import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os


@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    
    def __call__(self, results):
        occ = np.load(results['occ_path'])
        occ = occ.astype(np.float32)
        
        # class 0 is 'ignore' class
        if self.use_semantic:
            occ[..., 3][occ[..., 3] == 0] = 255
        else:
            occ = occ[occ[..., 3] > 0]
            occ[..., 3] = 1
        
        results['gt_occ'] = occ
        
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

