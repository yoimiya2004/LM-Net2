import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only


class BaseModel():
    """模型基类，所有模型都继承自此基类，提供通用的模型操作功能"""

    def __init__(self, opt):
        """初始化基类

        Args:
            opt (dict): 配置参数字典，包含训练/测试相关的所有配置
        """
        self.opt = opt  # 保存配置参数
        # 根据GPU数量选择设备（cuda或cpu）
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']  # 标记是否为训练模式
        self.schedulers = []  # 学习率调度器列表
        self.optimizers = []  # 优化器列表

    def feed_data(self, data):
        """喂入数据到模型，需在子类中实现具体逻辑"""
        pass

    def optimize_parameters(self):
        """优化模型参数，需在子类中实现具体逻辑"""
        pass

    def get_current_visuals(self):
        """获取当前的可视化结果，需在子类中实现具体逻辑"""
        pass

    def save(self, epoch, current_iter):
        """保存网络和训练状态

        Args:
            epoch (int): 当前轮次
            current_iter (int): 当前迭代次数
        """
        pass

    def validation(self, dataloader, current_iter, epoch, tb_logger, save_img=False, save_as_dir=None):
        """验证函数，根据是否分布式训练调用不同的验证方法

        Args:
            dataloader (torch.utils.data.DataLoader): 验证数据集加载器
            current_iter (int): 当前迭代次数
            epoch (int): 当前轮次
            tb_logger (tensorboard logger): Tensorboard日志记录器
            save_img (bool): 是否保存图像，默认False
            save_as_dir (str): 图像保存路径，默认None
        """
        if self.opt['dist']:
            # 分布式验证
            self.dist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
        else:
            # 非分布式验证
            self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    def _initialize_best_metric_results(self, dataset_name):
        """初始化最佳指标结果字典，用于记录最佳指标值和对应的迭代次数

        Args:
            dataset_name (str): 数据集名称
        """
        # 如果已初始化该数据集的指标，则直接返回
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        # 如果还没有最佳指标字典，则创建
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # 为每个数据集添加记录项
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            # 指标优化方向（更高更好或更低更好）
            better = content.get('better', 'higher')
            # 初始值（负无穷或正无穷，根据优化方向）
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_metric_result(self, dataset_name, metric, val, current_iter):
        """更新指标结果（不判断是否为最佳）

        Args:
            dataset_name (str): 数据集名称
            metric (str): 指标名称
            val (float): 指标值
            current_iter (int): 当前迭代次数
        """
        self.best_metric_results[dataset_name][metric]['val'] = val
        self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        """更新最佳指标结果（判断是否为最佳并更新）

        Args:
            dataset_name (str): 数据集名称
            metric (str): 指标名称
            val (float): 指标值
            current_iter (int): 当前迭代次数

        Returns:
            bool: 是否更新了最佳指标
        """
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            # 更高更好的指标
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
                return True
            else:
                return False
        else:
            # 更低更好的指标
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
                return True
            else:
                return False

    def model_ema(self, decay=0.999):
        """模型指数移动平均（EMA）更新，用于提高模型稳定性

        Args:
            decay (float): 衰减系数，默认0.999
        """
        net_g = self.get_bare_model(self.net_g)  # 获取原始模型（去除分布式包装）

        # 获取当前模型和EMA模型的参数字典
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        # 更新EMA模型参数
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def copy_model(self, net_a, net_b):
        """复制模型参数（从net_a到net_b）

        Args:
            net_a (nn.Module): 源模型
            net_b (nn.Module): 目标模型
        """
        tmp_net_a = self.get_bare_model(net_a)  # 获取原始模型
        tmp_net_b = self.get_bare_model(net_b)
        tmp_net_b.load_state_dict(tmp_net_a.state_dict())  # 复制参数

    def get_current_log(self):
        """获取当前的日志字典"""
        return self.log_dict

    def model_to_device(self, net):
        """将模型移动到指定设备，并根据配置包装分布式或数据并行

        Args:
            net (nn.Module): 要处理的网络

        Returns:
            nn.Module: 处理后的网络（已移动到设备并可能包装并行）
        """
        net = net.to(self.device)  # 移动到设备
        if self.opt['dist']:
            # 分布式训练：使用DistributedDataParallel包装
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            # 多GPU但非分布式：使用DataParallel包装
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        """创建优化器

        Args:
            optim_type (str): 优化器类型
            params (iterable): 要优化的参数
            lr (float): 学习率
            **kwargs: 优化器的其他参数

        Returns:
            torch.optim.Optimizer: 创建的优化器
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr,** kwargs)
        else:
            raise NotImplementedError(f'优化器 {optim_type} 尚未支持')
        return optimizer

    def setup_schedulers(self):
        """设置学习率调度器，根据配置创建相应的调度器"""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')  # 获取调度器类型
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            # 多步学习率调度器
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            # 余弦退火重启调度器
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer,** train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            # 循环余弦退火重启调度器
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'调度器 {scheduler_type} 尚未实现')

    def get_bare_model(self, net):
        """获取原始模型（去除DataParallel或DistributedDataParallel包装）

        Args:
            net (nn.Module): 可能被包装的网络

        Returns:
            nn.Module: 原始网络
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """打印网络结构和参数数量（仅主进程执行）

        Args:
            net (nn.Module): 要打印的网络
        """
        # 获取网络类名（处理包装情况）
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)  # 获取原始网络
        net_str = str(net)  # 网络结构字符串
        net_params = sum(map(lambda x: x.numel(), net.parameters()))  # 总参数数量

        logger = get_root_logger()
        logger.info(f'网络: {net_cls_str}, 参数数量: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """设置学习率（用于热身阶段）

        Args:
            lr_groups_l (list): 学习率组列表，每个元素对应一个优化器的学习率组
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """获取初始学习率（由调度器设置）

        Returns:
            list: 每个优化器的初始学习率组
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """更新学习率（支持热身阶段）

        Args:
            current_iter (int): 当前迭代次数
            warmup_iter (int): 热身迭代次数，-1表示不使用热身，默认-1
        """
        if current_iter > 1:
            # 非热身阶段，由调度器更新学习率
            for scheduler in self.schedulers:
                scheduler.step()
        # 处理热身阶段学习率
        if current_iter < warmup_iter:
            # 获取初始学习率
            init_lr_g_l = self._get_init_lr()
            # 线性热身学习率计算
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # 设置热身学习率
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        """获取当前学习率

        Returns:
            list: 每个优化器的当前学习率
        """
        return [optim.param_groups[0]['lr'] for optim in self.optimizers]

    @master_only
    def save_network(self, net, net_label, current_iter, epoch, param_key='params'):
        """保存网络参数（仅主进程执行）

        Args:
            net (nn.Module | list[nn.Module]): 要保存的网络（或网络列表）
            net_label (str): 网络标签（如'net_g'）
            current_iter (int): 当前迭代次数，-1表示最新
            epoch (int): 当前轮次
            param_key (str | list[str]): 保存参数的键名，默认'params'
        """
        if current_iter == -1:
            current_iter = 'latest'  # 用'latest'表示最新模型
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)  # 保存路径

        # 处理网络和参数键为列表的情况
        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), '网络和参数键的数量必须相同'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)  # 获取原始网络
            state_dict = net_.state_dict()  # 获取参数字典
            # 去除可能的'module.'前缀
            for key, param in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                state_dict[key] = param.cpu()  # 移动到CPU保存
            save_dict[param_key_] = state_dict
        save_dict['iter'] = current_iter  # 记录迭代次数
        save_dict['epoch'] = epoch  # 记录轮次

        # 处理可能的保存错误，重试机制
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'保存模型错误: {e}, 剩余重试次数: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'仍然无法保存 {save_path}，忽略此错误')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """打印加载模型时的不同键（名称不同或尺寸不同）

        1. 打印名称不同的键
        2. 若strict=False，打印名称相同但尺寸不同的键（这些键将被忽略）

        Args:
            crt_net (torch model): 当前网络
            load_net (dict): 加载的网络参数
            strict (bool): 是否严格匹配，默认True
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()  # 当前网络参数字典
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        # 打印名称不同的键
        if crt_net_keys != load_net_keys:
            logger.warning('当前网络有而加载网络没有的键:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('加载网络有而当前网络没有的键:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # 检查名称相同但尺寸不同的键（仅在非严格模式下）
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'尺寸不同，忽略键 [{k}]: 当前网络: '
                                   f'{crt_net[k].shape}; 加载网络: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)  # 标记为忽略

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """加载网络参数

        Args:
            net (nn.Module): 要加载参数的网络
            load_path (str): 参数文件路径
            strict (bool): 是否严格匹配，默认True
            param_key (str): 参数在文件中的键名，默认'params'
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)  # 获取原始网络
        # 加载参数文件（映射到当前设备）
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            # 检查参数键是否存在，不存在则尝试'params'
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('加载: params_ema不存在，使用params')
            load_net = load_net[param_key]  # 提取指定键的参数
        logger.info(f'从 {load_path} 加载 {net.__class__.__name__} 模型，参数键: [{param_key}]')
        # 去除可能的'module.'前缀
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        # 打印不同的键
        self._print_different_keys_loading(net, load_net, strict)
        # 加载参数
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """保存训练状态（用于断点续训），仅主进程执行

        Args:
            epoch (int): 当前轮次
            current_iter (int): 当前迭代次数
        """
        if current_iter != -1:
            # 构建状态字典
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            # 保存优化器状态
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            # 保存调度器状态
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # 处理可能的保存错误，重试机制
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'保存训练状态错误: {e}, 剩余重试次数: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'仍然无法保存 {save_path}，忽略此错误')

    def resume_training(self, resume_state):
        """恢复训练状态（从断点继续训练）

        Args:
            resume_state (dict): 要恢复的训练状态
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        # 检查优化器和调度器数量是否匹配
        assert len(resume_optimizers) == len(self.optimizers), '优化器数量不匹配'
        assert len(resume_schedulers) == len(self.schedulers), '调度器数量不匹配'
        # 恢复优化器状态
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        # 恢复调度器状态
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """归并损失字典（分布式训练时在不同GPU间平均损失）

        Args:
            loss_dict (OrderedDict): 损失字典

        Returns:
            OrderedDict: 归并后的损失字典（包含均值）
        """
        with torch.no_grad():
            if self.opt['dist']:
                # 分布式训练：收集所有GPU的损失并平均
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)  # 汇总到主进程
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']  # 主进程计算平均值
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            # 计算每个损失的均值并转换为标量
            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict


# import os
# import time
# import torch
# from collections import OrderedDict
# from copy import deepcopy
# from torch.nn.parallel import DataParallel, DistributedDataParallel
#
# from basicsr.models import lr_scheduler as lr_scheduler
# from basicsr.utils import get_root_logger
# from basicsr.utils.dist_util import master_only
#
#
# class BaseModel():
#     """Base model."""
#
#     def __init__(self, opt):
#         self.opt = opt
#         self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
#         self.is_train = opt['is_train']
#         self.schedulers = []
#         self.optimizers = []
#
#     def feed_data(self, data):
#         pass
#
#     def optimize_parameters(self):
#         pass
#
#     def get_current_visuals(self):
#         pass
#
#     def save(self, epoch, current_iter):
#         """Save networks and training state."""
#         pass
#
#     def validation(self, dataloader, current_iter, epoch, tb_logger, save_img=False, save_as_dir=None):
#         """Validation function.
#
#         Args:
#             dataloader (torch.utils.data.DataLoader): Validation dataloader.
#             current_iter (int): Current iteration.
#             tb_logger (tensorboard logger): Tensorboard logger.
#             save_img (bool): Whether to save images. Default: False.
#         """
#         if self.opt['dist']:
#             self.dist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
#         else:
#             self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
#
#     def _initialize_best_metric_results(self, dataset_name):
#         """Initialize the best metric results dict for recording the best metric value and iteration."""
#         if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
#             return
#         elif not hasattr(self, 'best_metric_results'):
#             self.best_metric_results = dict()
#
#         # add a dataset record
#         record = dict()
#         for metric, content in self.opt['val']['metrics'].items():
#             better = content.get('better', 'higher')
#             init_val = float('-inf') if better == 'higher' else float('inf')
#             record[metric] = dict(better=better, val=init_val, iter=-1)
#         self.best_metric_results[dataset_name] = record
#
#     def _update_metric_result(self, dataset_name, metric, val, current_iter):
#         self.best_metric_results[dataset_name][metric]['val'] = val
#         self.best_metric_results[dataset_name][metric]['iter'] = current_iter
#
#     def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
#         if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
#             if val >= self.best_metric_results[dataset_name][metric]['val']:
#                 self.best_metric_results[dataset_name][metric]['val'] = val
#                 self.best_metric_results[dataset_name][metric]['iter'] = current_iter
#                 return True
#             else:
#                 return False
#         else:
#             if val <= self.best_metric_results[dataset_name][metric]['val']:
#                 self.best_metric_results[dataset_name][metric]['val'] = val
#                 self.best_metric_results[dataset_name][metric]['iter'] = current_iter
#                 return True
#             else:
#                 return False
#
#     def model_ema(self, decay=0.999):
#         net_g = self.get_bare_model(self.net_g)
#
#         net_g_params = dict(net_g.named_parameters())
#         net_g_ema_params = dict(self.net_g_ema.named_parameters())
#
#         for k in net_g_ema_params.keys():
#             net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)
#
#     def copy_model(self, net_a, net_b):
#         """copy model from net_a to net_b"""
#         tmp_net_a = self.get_bare_model(net_a)
#         tmp_net_b = self.get_bare_model(net_b)
#         tmp_net_b.load_state_dict(tmp_net_a.state_dict())
#
#     def get_current_log(self):
#         return self.log_dict
#
#     def model_to_device(self, net):
#         """Model to device. It also warps models with DistributedDataParallel
#         or DataParallel.
#
#         Args:
#             net (nn.Module)
#         """
#         net = net.to(self.device)
#         if self.opt['dist']:
#             find_unused_parameters = self.opt.get('find_unused_parameters', False)
#             net = DistributedDataParallel(
#                 net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
#         elif self.opt['num_gpu'] > 1:
#             net = DataParallel(net)
#         return net
#
#     def get_optimizer(self, optim_type, params, lr, **kwargs):
#         if optim_type == 'Adam':
#             optimizer = torch.optim.Adam(params, lr, **kwargs)
#         else:
#             raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
#         return optimizer
#
#     def setup_schedulers(self):
#         """Set up schedulers."""
#         train_opt = self.opt['train']
#         scheduler_type = train_opt['scheduler'].pop('type')
#         if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
#             for optimizer in self.optimizers:
#                 self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
#         elif scheduler_type == 'CosineAnnealingRestartLR':
#             for optimizer in self.optimizers:
#                 self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
#         elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
#             for optimizer in self.optimizers:
#                 self.schedulers.append(lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
#         else:
#             raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
#
#     def get_bare_model(self, net):
#         """Get bare model, especially under wrapping with
#         DistributedDataParallel or DataParallel.
#         """
#         if isinstance(net, (DataParallel, DistributedDataParallel)):
#             net = net.module
#         return net
#
#     @master_only
#     def print_network(self, net):
#         """Print the str and parameter number of a network.
#
#         Args:
#             net (nn.Module)
#         """
#         if isinstance(net, (DataParallel, DistributedDataParallel)):
#             net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
#         else:
#             net_cls_str = f'{net.__class__.__name__}'
#
#         net = self.get_bare_model(net)
#         net_str = str(net)
#         net_params = sum(map(lambda x: x.numel(), net.parameters()))
#
#         logger = get_root_logger()
#         logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
#         logger.info(net_str)
#
#     def _set_lr(self, lr_groups_l):
#         """Set learning rate for warmup.
#
#         Args:
#             lr_groups_l (list): List for lr_groups, each for an optimizer.
#         """
#         for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
#             for param_group, lr in zip(optimizer.param_groups, lr_groups):
#                 param_group['lr'] = lr
#
#     def _get_init_lr(self):
#         """Get the initial lr, which is set by the scheduler.
#         """
#         init_lr_groups_l = []
#         for optimizer in self.optimizers:
#             init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
#         return init_lr_groups_l
#
#     def update_learning_rate(self, current_iter, warmup_iter=-1):
#         """Update learning rate.
#
#         Args:
#             current_iter (int): Current iteration.
#             warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
#                 Default： -1.
#         """
#         if current_iter > 1:
#             for scheduler in self.schedulers:
#                 scheduler.step()
#         # set up warm-up learning rate
#         if current_iter < warmup_iter:
#             # get initial lr for each group
#             init_lr_g_l = self._get_init_lr()
#             # modify warming-up learning rates
#             # currently only support linearly warm up
#             warm_up_lr_l = []
#             for init_lr_g in init_lr_g_l:
#                 warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
#             # set learning rate
#             self._set_lr(warm_up_lr_l)
#
#     def get_current_learning_rate(self):
#         return [optim.param_groups[0]['lr'] for optim in self.optimizers]
#
#     @master_only
#     def save_network(self, net, net_label, current_iter, epoch, param_key='params'):
#         """Save networks.
#
#         Args:
#             net (nn.Module | list[nn.Module]): Network(s) to be saved.
#             net_label (str): Network label.
#             current_iter (int): Current iter number.
#             param_key (str | list[str]): The parameter key(s) to save network.
#                 Default: 'params'.
#         """
#         if current_iter == -1:
#             current_iter = 'latest'
#         save_filename = f'{net_label}_{current_iter}.pth'
#         save_path = os.path.join(self.opt['path']['models'], save_filename)
#
#         net = net if isinstance(net, list) else [net]
#         param_key = param_key if isinstance(param_key, list) else [param_key]
#         assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'
#
#         save_dict = {}
#         for net_, param_key_ in zip(net, param_key):
#             net_ = self.get_bare_model(net_)
#             state_dict = net_.state_dict()
#             for key, param in state_dict.items():
#                 if key.startswith('module.'):  # remove unnecessary 'module.'
#                     key = key[7:]
#                 state_dict[key] = param.cpu()
#             save_dict[param_key_] = state_dict
#         save_dict['iter'] = current_iter
#         save_dict['epoch'] = epoch
#
#         # avoid occasional writing errors
#         retry = 3
#         while retry > 0:
#             try:
#                 torch.save(save_dict, save_path)
#             except Exception as e:
#                 logger = get_root_logger()
#                 logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
#                 time.sleep(1)
#             else:
#                 break
#             finally:
#                 retry -= 1
#         if retry == 0:
#             logger.warning(f'Still cannot save {save_path}. Just ignore it.')
#             # raise IOError(f'Cannot save {save_path}.')
#
#     def _print_different_keys_loading(self, crt_net, load_net, strict=True):
#         """Print keys with different name or different size when loading models.
#
#         1. Print keys with different names.
#         2. If strict=False, print the same key but with different tensor size.
#             It also ignore these keys with different sizes (not load).
#
#         Args:
#             crt_net (torch model): Current network.
#             load_net (dict): Loaded network.
#             strict (bool): Whether strictly loaded. Default: True.
#         """
#         crt_net = self.get_bare_model(crt_net)
#         crt_net = crt_net.state_dict()
#         crt_net_keys = set(crt_net.keys())
#         load_net_keys = set(load_net.keys())
#
#         logger = get_root_logger()
#         if crt_net_keys != load_net_keys:
#             logger.warning('Current net - loaded net:')
#             for v in sorted(list(crt_net_keys - load_net_keys)):
#                 logger.warning(f'  {v}')
#             logger.warning('Loaded net - current net:')
#             for v in sorted(list(load_net_keys - crt_net_keys)):
#                 logger.warning(f'  {v}')
#
#         # check the size for the same keys
#         if not strict:
#             common_keys = crt_net_keys & load_net_keys
#             for k in common_keys:
#                 if crt_net[k].size() != load_net[k].size():
#                     logger.warning(f'Size different, ignore [{k}]: crt_net: '
#                                    f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
#                     load_net[k + '.ignore'] = load_net.pop(k)
#
#     def load_network(self, net, load_path, strict=True, param_key='params'):
#         """Load network.
#
#         Args:
#             load_path (str): The path of networks to be loaded.
#             net (nn.Module): Network.
#             strict (bool): Whether strictly loaded.
#             param_key (str): The parameter key of loaded network. If set to
#                 None, use the root 'path'.
#                 Default: 'params'.
#         """
#         logger = get_root_logger()
#         net = self.get_bare_model(net)
#         load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
#         if param_key is not None:
#             if param_key not in load_net and 'params' in load_net:
#                 param_key = 'params'
#                 logger.info('Loading: params_ema does not exist, use params.')
#             load_net = load_net[param_key]
#         logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
#         # remove unnecessary 'module.'
#         for k, v in deepcopy(load_net).items():
#             if k.startswith('module.'):
#                 load_net[k[7:]] = v
#                 load_net.pop(k)
#         self._print_different_keys_loading(net, load_net, strict)
#         net.load_state_dict(load_net, strict=strict)
#
#     @master_only
#     def save_training_state(self, epoch, current_iter):
#         """Save training states during training, which will be used for
#         resuming.
#
#         Args:
#             epoch (int): Current epoch.
#             current_iter (int): Current iteration.
#         """
#         if current_iter != -1:
#             state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
#             for o in self.optimizers:
#                 state['optimizers'].append(o.state_dict())
#             for s in self.schedulers:
#                 state['schedulers'].append(s.state_dict())
#             save_filename = f'{current_iter}.state'
#             save_path = os.path.join(self.opt['path']['training_states'], save_filename)
#
#             # avoid occasional writing errors
#             retry = 3
#             while retry > 0:
#                 try:
#                     torch.save(state, save_path)
#                 except Exception as e:
#                     logger = get_root_logger()
#                     logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
#                     time.sleep(1)
#                 else:
#                     break
#                 finally:
#                     retry -= 1
#             if retry == 0:
#                 logger.warning(f'Still cannot save {save_path}. Just ignore it.')
#                 # raise IOError(f'Cannot save {save_path}.')
#
#     def resume_training(self, resume_state):
#         """Reload the optimizers and schedulers for resumed training.
#
#         Args:
#             resume_state (dict): Resume state.
#         """
#         resume_optimizers = resume_state['optimizers']
#         resume_schedulers = resume_state['schedulers']
#         assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
#         assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
#         for i, o in enumerate(resume_optimizers):
#             self.optimizers[i].load_state_dict(o)
#         for i, s in enumerate(resume_schedulers):
#             self.schedulers[i].load_state_dict(s)
#
#     def reduce_loss_dict(self, loss_dict):
#         """reduce loss dict.
#
#         In distributed training, it averages the losses among different GPUs .
#
#         Args:
#             loss_dict (OrderedDict): Loss dict.
#         """
#         with torch.no_grad():
#             if self.opt['dist']:
#                 keys = []
#                 losses = []
#                 for name, value in loss_dict.items():
#                     keys.append(name)
#                     losses.append(value)
#                 losses = torch.stack(losses, 0)
#                 torch.distributed.reduce(losses, dst=0)
#                 if self.opt['rank'] == 0:
#                     losses /= self.opt['world_size']
#                 loss_dict = {key: loss for key, loss in zip(keys, losses)}
#
#             log_dict = OrderedDict()
#             for name, value in loss_dict.items():
#                 log_dict[name] = value.mean().item()
#
#             return log_dict
