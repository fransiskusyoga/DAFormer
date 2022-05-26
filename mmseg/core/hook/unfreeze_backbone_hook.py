from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class UnfreezeBackboneIterBasedHookMIT(Hook):
    """Unfreeze backbone network Hook.

    Args:
        unfreeze_epoch (int): The epoch unfreezing the backbone network.
    """

    def __init__(self, unfreeze_iter=1):
        self.unfreeze_iter = unfreeze_iter

    def before_train_iter(self, runner):
        # Unfreeze the backbone network.
        # Only valid for mit_b0-mit_b5.
        if runner.iter == self.unfreeze_iter:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            if getattr(model, 'get_model') is not None:
                backbone = model.get_model().backbone
            else:
                backbone = model.backbone
            backbone.frozen_stages = -1
            for param in backbone.parameters():
                param.requires_grad = True
            
