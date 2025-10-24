from mmcv.runner import HOOKS, Hook
# import logging
from mmdet.utils import get_root_logger
import torch


@HOOKS.register_module()
class MergeWeightHook(Hook):
    """Custom hook to call `merge_correlated` every 1000 steps."""

    def __init__(self, merge_interval=100, warmup_iters=50, warmup_ratio=0.02, lora_key="lora_pw", prune_ratio=0.99):
        self.interval = merge_interval  # Interval in steps
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.current_warmup_iter = warmup_iters + 1
        self.base_lrs = {}
        self.lora_param_ids = set()
        self.lora_key=lora_key
        self.prune_ratio = prune_ratio
    
    def before_run(self, runner):
        """Store base learning rates before training starts."""
        logger = get_root_logger()
        if len(self.lora_param_ids) == 0:
            for name, param in runner.model.module.backbone.named_parameters(): # obtain lora params from backbone
                if self.lora_key in name:
                    self.lora_param_ids.add(id(param))
        logger.info("[LoRA params found ...., total: {}".format(len(self.lora_param_ids)))
        for param_group in runner.optimizer.param_groups:
            # self.base_lrs = {
            #     id(pg): pg['lr'] for pg in runner.optimizer.param_groups
            #     if any(id(p) in self.lora_param_ids for p in pg['params'])}
            self.base_lrs[id(param_group)] = param_group['lr']

    def after_train_iter(self, runner):
        """Called after every training iteration."""
        logger = get_root_logger()
        if (runner.iter ) % self.interval == 0 and runner.iter > 0:
            # merge weights
            runner.model.module.backbone.merge_correlated(merge_weights=True)
            self.reset_optimizer(runner.optimizer)
            self.current_warmup_iter = 0 # reset warmup
            self.before_run(runner) # obtain learning rate
        
        if self.current_warmup_iter < self.warmup_iters:
            logger.info("[Warm up lora ...], iter: {}".format(self.current_warmup_iter))
            self.apply_warmup(runner.optimizer)
            self.current_warmup_iter += 1
            # for param_group in runner.optimizer.param_groups:
            #     logger.info('[param lr: {}]'.format(param_group['lr']))
            # logger.info('='*30)
    
    @torch.no_grad()
    def magnitude_pruning_(self, tensor, prune_ratio):
        """
        https://github.com/Guitaricet/relora/blob/main/peft_pretraining/training_utils.py#L161
        Performs magnitude pruning dimensionality reduction **inplace**.
        Only reduces the inner dimensionality, does not affect the shape of the tensor
        """
        tensor_magnitude = torch.abs(tensor)
        threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

        mask = tensor_magnitude > threshold
        tensor.mul_(mask.to(dtype=tensor.dtype)).add_(1e-9)

    def reset_optimizer(self, optimizer):
        """Reset optimizer states after merging."""
        reset_cnt = 0
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if id(p) in self.lora_param_ids:
                    if p in optimizer.state:
                        # ["exp_avg", "exp_avg_sq"]
                        # optimizer.state[p] = {}
                        self.magnitude_pruning_(optimizer.state[p]["exp_avg"], self.prune_ratio)
                        self.magnitude_pruning_(optimizer.state[p]["exp_avg_sq"], self.prune_ratio)
                        reset_cnt += 1
        logger = get_root_logger()
        logger.info("[Reset lora state in optimizer..., total: {}".format(reset_cnt))

    def apply_warmup(self, optimizer):
        """Gradually restore learning rate using linear warmup."""
        warmup_factor = self.warmup_ratio + (1 - self.warmup_ratio) * (self.current_warmup_iter / self.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.base_lrs[id(param_group)] * warmup_factor