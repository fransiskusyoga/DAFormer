#from .detr import DETR


from mmseg.models.detectors.detr import DETR
from mmseg.models.builder import SEGMENTORS
from mmseg.models.detectors.detr_plus import DETR_plus
@SEGMENTORS.register_module()
class PanSeg(DETR_plus):

    def __init__(self, *args, **kwargs):
        super(DETR_plus, self).__init__(*args, **kwargs)
        self.count=0
