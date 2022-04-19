from . import CityscapesDataset
from .builder import DATASETS
from .custom_panuda import CustomDatasetPanUDA
import json

@DATASETS.register_module()
class SynthiaDataset_panoptic(CustomDatasetPanUDA):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        if 'rare_class_sampling'in kwargs:
            kwargs.pop('rare_class_sampling')
        super(SynthiaDataset_panoptic, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)

