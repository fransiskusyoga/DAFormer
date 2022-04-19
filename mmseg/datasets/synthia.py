from . import CityscapesDataset
from .builder import DATASETS
from .custom_seguda import CustomDatasetSegUDA


@DATASETS.register_module()
class SynthiaDataset(CustomDatasetSegUDA):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        if 'rare_class_sampling'in kwargs:
            kwargs.pop('rare_class_sampling')
        super(SynthiaDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
