#from ..builder import DETECTORS
#from .detr import DETR


from mmdet.models.detectors.detr import DETR
from mmseg.models.builder import DETECTORS
from mmseg.models.detectors.detr_plus import DETR_plus
@DETECTORS.register_module()
class PanSeg(DETR_plus):

    def __init__(self, *args, **kwargs):
        super(DETR_plus, self).__init__(*args, **kwargs)
        self.count=0
