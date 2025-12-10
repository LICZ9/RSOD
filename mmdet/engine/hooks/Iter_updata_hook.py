from mmengine.hooks import Hook
from mmdet.registry import HOOKS
@HOOKS.register_module()
class IterUpdateHook(Hook):
    def before_train_iter(self, runner,batch_idx=None, **kwargs):
        """在每次训练迭代前更新模型中的当前迭代数"""
        runner.model.iter = runner.iter