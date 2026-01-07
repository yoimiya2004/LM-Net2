from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu
from torchvision import models
from basicsr.archs import build_network  # 从BasicSR框架构建网络构建工具
from basicsr.losses import build_loss  # 从BasicSR获取损失函数构建工具
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor  # 工具函数
from basicsr.utils.registry import MODEL_REGISTRY  # 模型注册器，用于注册自定义模型
from .base_model import BaseModel  # 基础模型类，提供通用功能
import copy

import pyiqa  # 图像质量评估库
from .cal_ssim import SSIM  # 自定义SSIM评估指标
from torch import nn
import sys


def Normalize(x):
    """归一化函数：将输入张量缩放到[0, 255]范围"""
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin  # 线性归一化公式


def dwt_init(x):
    """二维离散小波变换（DWT）实现，使用哈尔小波基
    输入: 图像张量 (B, C, H, W)
    输出: 小波分解后的4个子带 (4B, C, H/2, W/2)，按LL, HL, LH, HH顺序拼接
    """
    # 对图像进行2x2分块采样
    x01 = x[:, :, 0::2, :] / 2  # 取偶数行（0,2,4...）
    x02 = x[:, :, 1::2, :] / 2  # 取奇数行（1,3,5...）
    x1 = x01[:, :, :, 0::2]  # 取偶数列（LL子带初步）
    x2 = x02[:, :, :, 0::2]  # HL子带初步
    x3 = x01[:, :, :, 1::2]  # LH子带初步
    x4 = x02[:, :, :, 1::2]  # HH子带初步

    # 计算四个子带
    x_LL = x1 + x2 + x3 + x4  # 低频分量（近似分量）
    x_HL = -x1 - x2 + x3 + x4  # 水平高频分量
    x_LH = -x1 + x2 - x3 + x4  # 垂直高频分量
    x_HH = x1 - x2 - x3 + x4  # 对角线高频分量

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)  # 在批次维度拼接


def iwt_init(x):
    """逆离散小波变换（IWT），重构原始图像
    输入: 小波分解后的张量 (4B, C, H/2, W/2)
    输出: 重构图像 (B, C, H, W)
    """
    r = 2  # 上采样倍数
    in_batch, in_channel, in_height, in_width = x.size()
    # 计算输出尺寸
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    # 分离四个子带
    x1 = x[0:out_batch, :, :] / 2  # LL子带
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2  # HL子带
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2  # LH子带
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2  # HH子带

    # 初始化重构图像张量
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    # 填充像素（逆过程：将子带数据还原到原始位置）
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    """小波变换层封装（无参数，仅前向计算）"""

    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 无需梯度更新

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    """逆小波变换层封装（无参数，仅前向计算）"""

    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False  # 无需梯度更新

    def forward(self, x):
        return iwt_init(x)


class Vgg19(torch.nn.Module):
    """VGG19特征提取器，用于感知损失计算
    截取预训练VGG19的前5个卷积块输出，作为不同层级的特征
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # 加载预训练VGG19的特征层
        vgg_pretrained_features = models.vgg19(pretrained=True).cuda().features
        # 分割不同卷积块（按ReLU激活层划分）
        self.slice1 = torch.nn.Sequential()  # 第1个卷积块（前2层）
        self.slice2 = torch.nn.Sequential()  # 第2个卷积块（2-7层）
        self.slice3 = torch.nn.Sequential()  # 第3个卷积块（7-12层）
        self.slice4 = torch.nn.Sequential()  # 第4个卷积块（12-21层）
        self.slice5 = torch.nn.Sequential()  # 第5个卷积块（21-30层）

        # 填充各卷积块的层
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # 冻结参数（仅用于特征提取，不参与训练）
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """提取多尺度特征"""
        h_relu1 = self.slice1(X)  # 第1层特征（低层级，边缘/纹理）
        h_relu2 = self.slice2(h_relu1)  # 第2层特征
        h_relu3 = self.slice3(h_relu2)  # 第3层特征
        h_relu4 = self.slice4(h_relu3)  # 第4层特征（高层级，语义）
        h_relu5 = self.slice5(h_relu4)  # 第5层特征
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    """对比损失：拉近输出与GT的特征距离，拉远输出与退化输入/中间结果的距离
    基于VGG特征的对比学习
    """

    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()  # 初始化VGG特征提取器
        self.l1 = nn.L1Loss()  # L1损失计算特征距离
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # 各层特征权重（低层权重小，高层权重大）
        self.ab = ablation  # 是否为消融实验模式（仅用正样本损失）

    def forward(self, a, p, n):
        """
        a: 待优化的输出（anchor）
        p: 正样本（GT）
        n: 负样本（退化输入或中间结果）
        """
        # 提取三者的VGG特征
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        for i in range(len(a_vgg)):
            # 计算anchor与正样本的距离
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())  # detach避免GT特征参与梯度计算
            if not self.ab:
                # 计算anchor与负样本的距离，总损失为d_ap/(d_an + 1e-7)
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                # 消融模式：仅用正样本损失
                contrastive = d_ap

            # 加权累加各层损失
            loss += self.weights[i] * contrastive
        return loss


@MODEL_REGISTRY.register()  # 注册模型到BasicSR框架
class FeMaSRModel(BaseModel):
    """FeMaSR模型主类，继承自BaseModel，实现图像去雾/增强功能"""

    def __init__(self, opt):
        super().__init__(opt)  # 调用父类初始化

        # 构建生成网络（WaveMamba）
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)  # 移动到指定设备（GPU/CPU）

        # 初始化评估指标和损失函数
        self.ssim = SSIM().cuda()  # SSIM评估指标
        self.l1 = nn.L1Loss().cuda()  # L1损失
        self.LL = None  # 存储小波低频分量（用于损失计算）
        self.x_stage1 = None  # 存储网络中间输出（用于对比损失）
        self.cr = ContrastLoss()  # 初始化对比损失

        # 定义验证指标（如PSNR、LPIPS等）
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                # 创建评估指标实例（如pyiqa的psnr、lpips等）
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # 加载预训练HQ模型（如果配置了LQ_stage）
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'LQ阶段需指定HQ预训练模型路径'
            self.load_network(self.net_g, load_path, False)  # 加载模型，不严格匹配参数

        # 加载生成网络预训练权重
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'从 {load_path} 加载生成网络权重')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        # 训练模式下的初始化
        if self.is_train:
            self.init_training_settings()  # 初始化优化器、损失函数等
        self.net_g_best = copy.deepcopy(self.net_g)  # 保存最佳模型副本

    def init_training_settings(self):
        """初始化训练相关配置：损失函数、优化器、调度器等"""
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()  # 设置为训练模式

        # 定义像素损失（如L1、L2）
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # 定义频域损失（如FFT损失）
        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # 设置优化器和学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """配置优化器（仅优化生成网络）"""
        train_opt = self.opt['train']
        optim_params = []
        # 收集所有可训练参数
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'参数 {k} 不参与优化')

        # 初始化生成器优化器（如Adam）
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """加载输入数据到设备"""
        self.lq = data['lq'].to(self.device)  # 低质量输入（如含雾图像）
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)  # 高质量标签（如无雾图像）

    def optimize_parameters(self, current_iter):
        """核心训练逻辑：前向传播计算损失，反向传播更新参数"""
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()  # 清空梯度

        # 前向传播：获取网络输出（小波低频分量、中间结果、最终输出）
        self.LL, self.x_stage1, self.output = self.net_g(self.lq)

        l_g_total = 0  # 总损失
        loss_dict = OrderedDict()  # 记录各损失项

        # 1. 计算像素损失（输出与GT的L1+SSIM损失）
        l_pix = self.l1(self.output, self.gt) + (1 - self.ssim(self.output, self.gt)) * 0.1
        l_g_total += l_pix
        loss_dict['l_pix'] = l_pix

        # 2. 计算小波低频分量损失（网络输出的LL子带与GT的LL子带对齐）
        dwt = DWT()
        n, c, h, w = self.gt.shape
        gt_dwt = dwt(self.gt)  # 对GT做小波变换
        gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]  # 分离LL子带和高频子带
        l_LL = self.l1(self.LL, gt_LL)  # LL子带L1损失
        l_g_total += l_LL
        loss_dict['l_LL'] = l_LL

        # 3. 计算对比损失（拉近输出与GT，拉远输出与输入/中间结果）
        # 对比损失1：输出 vs GT vs 原始输入；对比损失2：输出 vs GT vs 中间结果
        l_cr = 0.75 * self.cr(self.output, self.gt, self.lq) + 1.25 * self.cr(self.output, self.gt, self.x_stage1)
        l_g_total += 0.001 * l_cr  # 对比损失权重
        loss_dict['l_cr'] = l_cr

        # 4. 计算频域损失（如FFT损失，可选）
        if train_opt.get('fft_opt', None):
            l_fft = self.cri_fft(self.output, self.gt)
            l_g_total += l_fft
            loss_dict['l_freq'] = l_fft

        # 反向传播与参数更新
        l_g_total.mean().backward()  # 计算梯度
        self.optimizer_g.step()  # 更新参数

        self.log_dict = self.reduce_loss_dict(loss_dict)  # 记录损失（支持分布式训练）

    def test(self):
        """测试/推理逻辑：输入低质量图像，输出恢复结果"""
        self.net_g.eval()  # 设置为评估模式
        net_g = self.get_bare_model(self.net_g)  # 获取原始模型（去除DataParallel包装）
        min_size = 8000 * 8000  # 图像尺寸阈值，超过则使用分块推理（避免显存溢出）
        lq_input = self.lq
        _, _, h, w = lq_input.shape

        # 根据图像尺寸选择推理方式
        if h * w < min_size:
            # 整图推理
            self.output = net_g.test(lq_input)
        else:
            # 分块推理（大图像）
            self.output = net_g.test_tile(lq_input)

        self.net_g.train()  # 恢复训练模式

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir):
        """非分布式验证逻辑：计算指标、保存结果、更新最佳模型"""
        dataset_name = 'NTIRE2024'  # 验证数据集名称
        with_metrics = self.opt['val'].get('metrics') is not None  # 是否计算指标
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        pbar = tqdm(total=len(dataloader), unit='image')  # 进度条

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)  # 初始化最佳指标记录
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')  # 关键指标（如PSNR）

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]  # 获取图像名称
            self.feed_data(val_data)  # 加载数据
            self.test()  # 推理

            # 转换输出为图像格式
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]  # 准备指标计算数据

            # 释放显存
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # 保存输出图像
            if save_img:
                # 构建保存路径
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results', f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    suffix = self.opt['val']['suffix'] if self.opt['val']['suffix'] else self.opt['name']
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{suffix}.png')
                if save_as_dir:
                    imwrite(sr_img, osp.join(save_as_dir, f'{img_name}.png'))
                imwrite(sr_img, save_img_path)

            # 计算评估指标
            if with_metrics:
                for name, _ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += self.metric_funcs[name](*metric_data).item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        # 计算平均指标并更新最佳模型
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)  # 平均指标

            # 根据关键指标更新最佳模型
            if self.key_metric is not None:
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)
                if to_update:
                    self.copy_model(self.net_g, self.net_g_best)  # 更新最佳模型
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)  # 保存最佳模型

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)  # 记录日志

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """记录验证指标日志"""
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)

    def get_current_visuals(self):
        """获取当前可视化结果（用于训练过程中的可视化）"""
        vis_samples = 16  # 可视化样本数
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]  # 低质量输入
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]  # 模型输出
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]  # 高质量标签
        return out_dict

    def save(self, epoch, current_iter):
        """保存模型权重和训练状态"""
        self.save_network(self.net_g, 'net_g', current_iter, epoch)  # 保存生成网络
        self.save_training_state(epoch, current_iter)  # 保存训练状态（优化器、调度器等）

#
# from collections import OrderedDict
# from os import path as osp
# from tqdm import tqdm
#
# import torch
# import torchvision.utils as tvu
# from torchvision import models
# from basicsr.archs import build_network
# from basicsr.losses import build_loss
# from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
# from basicsr.utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel
# import copy
#
# import pyiqa
# from .cal_ssim import SSIM
# from torch import nn
# import sys
#
# def Normalize(x):
#     ymax = 255
#     ymin = 0
#     xmax = x.max()
#     xmin = x.min()
#     return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
#
#
# def dwt_init(x):
#
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
#
#
# # 使用哈尔 haar 小波变换来实现二维离散小波
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
#     x1 = x[0:out_batch, :, :] / 2
#     x2 = x[out_batch:out_batch * 2, :, :, :] / 2
#     x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
#     x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height,
#                      out_width]).float().to(x.device)
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h
#
#
# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return dwt_init(x)
#
#
# class IWT(nn.Module):
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)
#
#
# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg_pretrained_features = models.vgg19(pretrained=True).cuda().features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#
# class ContrastLoss(nn.Module):
#     def __init__(self, ablation=False):
#
#         super(ContrastLoss, self).__init__()
#         self.vgg = Vgg19().cuda()
#         self.l1 = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
#         self.ab = ablation
#
#     def forward(self, a, p, n):
#         a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
#         loss = 0
#
#         d_ap, d_an = 0, 0
#         for i in range(len(a_vgg)):
#             d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
#             if not self.ab:
#                 d_an = self.l1(a_vgg[i], n_vgg[i].detach())
#                 contrastive = d_ap / (d_an + 1e-7)
#             else:
#                 contrastive = d_ap
#
#             loss += self.weights[i] * contrastive
#         return loss
#
#
# @MODEL_REGISTRY.register()
# class FeMaSRModel(BaseModel):
#     def __init__(self, opt):
#         super().__init__(opt)
#
#         # 定义网络
#         self.net_g = build_network(opt['network_g'])
#         self.net_g = self.model_to_device(self.net_g)
#         self.ssim = SSIM().cuda()
#         self.l1 = nn.L1Loss().cuda()
#         self.LL = None
#         self.x_stage1 = None
#         self.cr = ContrastLoss()
#
#         # 定义评价指标函数
#         if self.opt['val'].get('metrics') is not None:
#             self.metric_funcs = {}
#             for _, opt in self.opt['val']['metrics'].items():
#                 mopt = opt.copy()
#                 name = mopt.pop('type', None)
#                 mopt.pop('better', None)
#                 self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
#
#         # 加载预先训练的HQ ckpt、冻结解码器和码本
#         self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
#         if self.LQ_stage:
#             load_path = self.opt['path'].get('pretrain_network_hq', None)
#             assert load_path is not None, 'Need to specify hq prior model path in LQ stage'
#
#             # hq_opt = self.opt['network_g'].copy()
#             # hq_opt['LQ_stage'] = False
#             # self.net_hq = build_network(hq_opt)
#             # self.net_hq = self.model_to_device(self.net_hq)
#             # self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])
#
#             self.load_network(self.net_g, load_path, False)
#             # frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
#             # if frozen_module_keywords is not None:
#             #     for name, module in self.net_g.named_modules():
#             #         for fkw in frozen_module_keywords:
#             #             if fkw in name:
#             #                 for p in module.parameters():
#             #                     p.requires_grad = False
#             #                 break
#
#         # 加载预训练模型
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         # print('#########################################################################',load_path)
#         logger = get_root_logger()
#         if load_path is not None:
#             logger.info(f'Loading net_g from {load_path}')
#             self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
#
#         if self.is_train:
#             self.init_training_settings()
#             # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
#             # self.net_d_best = copy.deepcopy(self.net_d)
#
#         self.net_g_best = copy.deepcopy(self.net_g)
#
#     def init_training_settings(self):
#         logger = get_root_logger()
#         train_opt = self.opt['train']
#         self.net_g.train()
#
#         # define network net_d
#         # self.net_d = build_network(self.opt['network_d'])
#         # self.net_d = self.model_to_device(self.net_d)
#         # load pretrained d models
#         # load_path = self.opt['path'].get('pretrain_network_d', None)
#         # # print(load_path)
#         # if load_path is not None:
#         #     logger.info(f'Loading net_d from {load_path}')
#         #     self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
#
#         # self.net_d.train()
#
#         # 定义损失函数
#         if train_opt.get('pixel_opt'):
#             self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
#         else:
#             self.cri_pix = None
#
#         if train_opt.get('fft_opt'):
#             self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
#         else:
#             self.cri_fft = None
#
#         # if train_opt.get('perceptual_opt'):
#         #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
#         #     self.model_to_device(self.cri_perceptual)
#         # else:
#         #     self.cri_perceptual = None
#
#         # if train_opt.get('gan_opt'):
#         #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
#
#         # self.net_d_iters = train_opt.get('net_d_iters', 1)
#         # self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
#
#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()
#
#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#         for k, v in self.net_g.named_parameters():
#             optim_params.append(v)
#             if not v.requires_grad:
#                 logger = get_root_logger()
#                 logger.warning(f'Params {k} will not be optimized.')
#
#         # 定义优化器
#         optim_type = train_opt['optim_g'].pop('type')
#         optim_class = getattr(torch.optim, optim_type)
#         self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
#         self.optimizers.append(self.optimizer_g)
#
#         # optimizer d
#         # optim_type = train_opt['optim_d'].pop('type')
#         # optim_class = getattr(torch.optim, optim_type)
#         # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
#         # self.optimizers.append(self.optimizer_d)
#
#     def feed_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         # self.lq_equalize = data['lq_equalize'].to(self.device)
#
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)
#
#
#     def print_network(self, model):
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print(model)
#         print("The number of parameters: {}".format(num_params))
#
#     def optimize_parameters(self, current_iter):
#         train_opt = self.opt['train']
#
#         # for p in self.net_d.parameters():
#         #     p.requires_grad = False
#         self.optimizer_g.zero_grad()
#
#         self.LL, self.x_stage1, self.output = self.net_g(self.lq)
#
#         l_g_total = 0
#         loss_dict = OrderedDict()
#
#         dwt = DWT()
#         n, c, h, w = self.gt.shape
#
#         gt_dwt = dwt(self.gt)
#         gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
#
#         l_pix = self.l1(self.output, self.gt) + (1 - self.ssim(self.output, self.gt)) * 0.1
#         l_g_total += l_pix
#         loss_dict['l_pix'] = l_pix
#
#         l_LL = self.l1(self.LL, gt_LL)
#         l_g_total += l_LL
#         loss_dict['l_LL'] = l_LL
#
#         l_cr = 0.75 * self.cr(self.output, self.gt, self.lq) + 1.25 * self.cr(self.output, self.gt, self.x_stage1)
#
#         l_g_total += 0.001 * l_cr
#         loss_dict['l_cr'] = l_cr
#
#         if train_opt.get('fft_opt', None):
#             l_fft = self.cri_fft(self.output, self.gt)
#             l_g_total += l_fft
#             loss_dict['l_freq'] = l_fft
#
#         l_g_total.mean().backward()
#
#         self.optimizer_g.step()
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     def test(self):
#         self.net_g.eval()
#         net_g = self.get_bare_model(self.net_g)
#         min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
#         lq_input = self.lq
#         # restoration = self.net_g(self.lq)
#         _, _, h, w = lq_input.shape
#         if h * w < min_size:
#             # out_img, feature_degradation, self.output = self.net_g(self.lq, feature=feature_degradation)
#             self.output = net_g.test(lq_input)
#         else:
#             self.output = net_g.test_tile(lq_input)
#         self.net_g.train()
#
#     def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
#         logger = get_root_logger()
#         logger.info('Only support single GPU validation.')
#         self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
#
#     def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
#                            save_img, save_as_dir):
#         # dataset_name = dataloader.dataset.opt['name']
#         dataset_name = 'NTIRE2024'
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#
#         pbar = tqdm(total=len(dataloader), unit='image')
#
#         if with_metrics:
#             if not hasattr(self, 'metric_results'):  # only execute in the first run
#                 self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
#             # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
#             self._initialize_best_metric_results(dataset_name)
#
#             # zero self.metric_results
#             self.metric_results = {metric: 0 for metric in self.metric_results}
#             self.key_metric = self.opt['val'].get('key_metric')
#
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#             self.feed_data(val_data)
#             self.test()
#
#             sr_img = tensor2img(self.output)
#             metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]
#
#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()
#
#             if save_img:
#                 if self.opt['is_train']:
#                     save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
#                                              f'{current_iter}',
#                                              f'{img_name}.png')
#                 else:
#                     if self.opt['val']['suffix']:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
#                     else:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["name"]}.png')
#                 if save_as_dir:
#                     save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
#                     imwrite(sr_img, save_as_img_path)
#                 imwrite(sr_img, save_img_path)
#
#             if with_metrics:
#                 # calculate metrics
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_result = self.metric_funcs[name](*metric_data)
#                     self.metric_results[name] += tmp_result.item()
#
#             pbar.update(1)
#             pbar.set_description(f'Test {img_name}')
#
#         pbar.close()
#
#         if with_metrics:
#             # calculate average metric
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#
#             if self.key_metric is not None:
#                 # If the best metric is updated, update and save best model
#                 to_update = self._update_best_metric_result(dataset_name, self.key_metric,
#                                                             self.metric_results[self.key_metric], current_iter)
#
#                 if to_update:
#                     for name, opt_ in self.opt['val']['metrics'].items():
#                         self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
#                     # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
#             else:
#                 # update each metric separately
#                 updated = []
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
#                                                                   current_iter)
#                     updated.append(tmp_updated)
#                 # save best model if any metric is updated
#                 if sum(updated):
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', '')
#                     # self.save_network(self.net_d, 'net_d_best', '')
#
#             self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
#
#     def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
#         log_str = f'Validation {dataset_name}\n'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#             if hasattr(self, 'best_metric_results'):
#                 log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
#                             f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
#             log_str += '\n'
#
#         logger = get_root_logger()
#         logger.info(log_str)
# #        if tb_logger:
# #            for metric, value in self.metric_results.items():
# #                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
#
#     def vis_single_code(self, up_factor=2):
#         net_g = self.get_bare_model(self.net_g)
#         codenum = self.opt['network_g']['codebook_params'][0][1]
#         with torch.no_grad():
#             code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
#             code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
#             output_img = net_g.decode_indices(code_idx)
#             output_img = tvu.make_grid(output_img, nrow=32)
#
#         return output_img.unsqueeze(0)
#
#     def get_current_visuals(self):
#         vis_samples = 16
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
#         out_dict['result'] = self.output.detach().cpu()[:vis_samples]
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
#         return out_dict
#
#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter, epoch)
#         # self.save_network(self.net_d, 'net_d', current_iter, epoch)
#         self.save_training_state(epoch, current_iter)
