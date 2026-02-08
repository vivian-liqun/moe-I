import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Vanilla,ConcatFusion_Swin
from models.swin_transformer import SwinTransformer
import numpy as np


def modality_drop(x_rgb, x_depth, p, prob,args=None):

    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]
    # print(f"modality_drop 函数接收到的 prob: {prob}")

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        # prob = np.array((1 / 3, 1 / 3, 1 / 3))
        prob = prob
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]

    if args.use_video_frames != 1:
        pv = torch.repeat_interleave(p, args.use_video_frames, dim=0)
        # print(pv.shape)
        x_depth = x_depth * pv[:, 1]
    else:
        x_depth = x_depth * p[:, 1]

    return x_rgb, x_depth, p


def modality_drop_v(x_rgb, x_depth, p, args=None):
    # p=[1,1]
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 3, 1 / 3, 1 / 3))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]

    if args.use_video_frames != 1:
        pv = torch.repeat_interleave(p, args.use_video_frames, dim=0)
        # print(pv.shape)
        x_depth = x_depth * pv[:, 1]
    else:
        x_depth = x_depth * p[:, 1]
    p = p.squeeze()
    return x_rgb, x_depth, p


class SmartUpsample(nn.Module):
    """自适应学习型上采样模块"""

    def __init__(self, scale_factors):
        super().__init__()
        self.scale_factors = scale_factors
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 动态计算目标尺寸
        H, W = x.shape[2:]
        target_H = int(H * self.scale_factors[0])
        target_W = int(W * self.scale_factors[1])

        x = F.interpolate(x,
                          size=(target_H, target_W),
                          mode='bilinear',
                          align_corners=False)
        return self.conv(x)


class PreciseUpsampler(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(512, 512, 3, padding=1)  # 保持尺寸

    def forward(self, x):
        # 精确尺寸计算
        B, C, H, W = x.shape
        scale_h = self.target_size[0] / H
        scale_w = self.target_size[1] / W

        # 分步调整避免量化误差
        x = F.interpolate(x,
                          scale_factor=(scale_h, scale_w),
                          mode='bilinear',
                          align_corners=False)

        # 尺寸校验
        assert x.shape[2:] == self.target_size, \
            f"尺寸误差: 预期{self.target_size}, 实际{x.shape[2:]}"

        return self.conv(x)

def mask_modalities_with_zeros(x_rgb, x_ir, p):
    """
    使用0填充的方式对三个模态特征进行掩码处理

    参数:
    x_rgb: RGB模态特征, 形状为 [B, C, H, W]
    x_ir: 红外模态特征, 形状为 [B, C, H, W]
    x_depth: 深度模态特征, 形状为 [B, C, H, W]
    p: 指示向量, 形状为 [B, 3], 每个元素为0或1
        p[:,0]=1表示保留RGB模态，0表示掩码
        p[:,1]=1表示保留IR模态，0表示掩码
        p[:,2]=1表示保留Depth模态，0表示掩码

    返回:
    masked_rgb, masked_ir, masked_depth: 使用0填充后的特征张量
    """
    B = x_rgb.size(0)

    # 将p的每个维度扩展为与对应特征相同的形状[2,6](@ref)
    p_rgb = p[:, 0].view(B, 1, 1, 1).expand_as(x_rgb)
    p_ir = p[:, 1].view(B, 1, 1, 1).expand_as(x_ir)


    # 创建布尔掩码（True表示保留，False表示用0填充）[2,3](@ref)
    mask_rgb = p_rgb.bool()
    mask_ir = p_ir.bool()


    # 应用掩码：将需要掩码的位置填充为0[1,2](@ref)
    masked_rgb = x_rgb.masked_fill(~mask_rgb, 0.0)  # ~mask_rgb表示需要填充的位置
    masked_ir = x_ir.masked_fill(~mask_ir, 0.0)


    return masked_rgb, masked_ir

class AVClassifier_valid(nn.Module):
    def __init__(self, args):
        super(AVClassifier_valid, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(args, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p = [0, 0]
        self.args = args
        self.fc_out = nn.Linear(1024, 100)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.audio_upsampler = PreciseUpsampler(target_size=(7, 20))
        self.visual_expander = PreciseUpsampler(target_size=(7, 20))

    def forward(self, audio, visual,p):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        a_feature = a
        v_feature = v

        a_feature = self.audio_upsampler(a_feature)  # [B,512,7,20]
        v_feature = self.visual_expander(v_feature)  # [B,512,7,20]
        a, v=mask_modalities_with_zeros(a_feature, v_feature,p)

        layer0 = torch.cat((a, v), dim=1)
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        features = self.pooling(layer3)
        features = features.view(features.shape[0], -1)
        out = self.fc_out(features)

        return a, v, out, p, layer1, layer2, layer3


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(args, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p=[0,0]
        self.args = args
        self.fc_out = nn.Linear(1024, 100)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))


        self.audio_upsampler =  PreciseUpsampler(target_size=(7,20))
        self.visual_expander =  PreciseUpsampler(target_size=(7,20))

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)


        a_feature = a
        v_feature = v

        a_feature = self.audio_upsampler(a_feature)  # [B,512,7,20]
        v_feature = self.visual_expander(v_feature)  # [B,512,7,20]
        a, v, p = modality_drop_v(a_feature, v_feature, self.p, args=self.args)

        layer0 = torch.cat((a,v), dim=1)
        layer1= self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        features = self.pooling(layer3)
        features = features.view(features.shape[0], -1)
        out = self.fc_out(features)

        return a, v,out,p,layer1,layer2,layer3

class AVClassifier_teacher(nn.Module):
    def __init__(self, args):
        super(AVClassifier_teacher, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(args, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p = [1, 1]
        self.args = args
        self.fc_out = nn.Linear(1024, 100)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.audio_upsampler = PreciseUpsampler(target_size=(7, 20))
        self.visual_expander = PreciseUpsampler(target_size=(7, 20))

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        a_feature = a
        v_feature = v

        a_feature = self.audio_upsampler(a_feature)  # [B,512,7,20]
        v_feature = self.visual_expander(v_feature)  # [B,512,7,20]
        a, v, p = modality_drop_v(a_feature, v_feature, [1,1], args=self.args)
        # (_, C, H, W) = v.size()
        # B = a.size()[0]
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)
        #
        # a = F.adaptive_avg_pool2d(a, 1)
        # v = F.adaptive_avg_pool3d(v, 1)
        #
        # a = torch.flatten(a, 1)
        # v = torch.flatten(v, 1)
        layer0 = torch.cat((a, v), dim=1)
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        features = self.pooling(layer3)
        features = features.view(features.shape[0], -1)
        out = self.fc_out(features)

        return a, v,out,p,layer1,layer2,layer3

class ACVWithPrompt(nn.Module):
    """
    基于MULTModel的提示增强版本，在last_hs1, last_hs_proj, mu_dul位置添加可训练提示参数
    支持外部直接提供掩码作为输入，不在映射中的掩码不添加提示
    """

    def __init__(self, acv_instance, args):
        super(ACVWithPrompt, self).__init__()

        # 保存原始MULT模型实例
        self.acv_model = acv_instance

        # 冻结原始模型的所有参数
        for param in self.acv_model.parameters():
            param.requires_grad = False


        # 初始化提示参数
        self._init_prompt_parameters()

    def _init_prompt_parameters(self):
        """初始化7种缺失类型对应的提示参数"""
        self.num_prompt_types = 3  # 对应7种缺失组合

        # 为last_hs1添加提示参数 [7, 90, 50]
        self.prompt_layer1 = nn.Parameter(
            torch.randn(self.num_prompt_types, 1024,7,20)
        )

        # 为last_hs_proj添加提示参数 [7, 90, 50]
        self.prompt_layer2 = nn.Parameter(
            torch.randn(self.num_prompt_types, 1024,7,20)
        )

        # 为mu_dul添加提示参数 [7, 90, 50]
        self.prompt_layer3 = nn.Parameter(
            torch.randn(self.num_prompt_types, 1024,7,20)
        )

        # 缺失类型映射：将缺失模式映射到提示索引
        self.missing_type_mapping = {
            (1, 0): 0,
            (0, 1): 1,
            (1, 1): 2,

        }

    def get_prompt_indices(self, external_mask):
        """
        根据外部提供的掩码获取对应的提示索引
        不在映射中的返回None表示不添加提示

        Args:
            external_mask: 外部提供的掩码 [batch_size, 3]
        """
        batch_size = external_mask.size(0)
        prompt_indices = []

        for i in range(batch_size):
            # 将外部掩码转换为元组作为键
            key = (
                int(external_mask[i, 0].item()),  # 音频
                int(external_mask[i, 1].item()),  # 视觉
            )

            if key in self.missing_type_mapping:
                prompt_indices.append(self.missing_type_mapping[key])
            else:
                # 不在映射中，返回None表示不添加提示
                prompt_indices.append(None)

        return prompt_indices

    def apply_prompts(self, features, prompt_params, prompt_weights, prompt_indices):
        """
        将带权重的提示参数应用到特征上
        对于prompt_indices为None的样本，不添加提示

        Args:
            features: 原始特征 [batch_size, channels, seq_len]
            prompt_params: 提示参数 [num_prompt_types, channels, seq_len]
            prompt_weights: 权重矩阵 [1, 7]，7种提示类型的权重
            prompt_indices: 提示索引列表，可能包含None
        """
        enhanced_features = features.clone()  # 创建副本

        # 确保prompt_weights是PyTorch张量
        if not isinstance(prompt_weights, torch.Tensor):
            prompt_weights = torch.tensor(prompt_weights, dtype=torch.float32, device=features.device)

        # 调整权重形状以便广播 [1, 7] -> [7, 1]
        weights_reshaped = prompt_weights.view(3, 1,1,1)

        # 为每个需要提示的样本添加带权重的提示
        for i, idx in enumerate(prompt_indices):
            if idx is not None:  # 只在有有效索引时添加提示
                # 获取对应的提示参数和权重
                prompt = prompt_params[idx]  # [channels, seq_len]
                weight = weights_reshaped[idx]  # [1, 1]

                # 提示参数乘以权重后添加到特征
                weighted_prompt = prompt * weight
                enhanced_features[i] = enhanced_features[i] + weighted_prompt

        return enhanced_features

    def forward(self, x_a, x_v, external_mask, prompt_weights):
        """
        前向传播，使用外部提供的掩码和权重
        不在映射中的掩码不添加提示

        Args:
            x_l: 语言模态输入 [batch_size, seq_len, 300]
            x_a: 音频模态输入 [batch_size, seq_len, 74]
            x_v: 视觉模态输入 [batch_size, seq_len, 35]
            external_mask: 外部提供的掩码 [batch_size, 3]
            prompt_weights: 权重矩阵 [1, 7]，7种提示类型的权重
        """
        batch_size = x_a.size(0)


        a, v,out,p,layer1,layer2,layer3 = self.acv_model( x_a, x_v)

        # 根据外部掩码获取提示索引（可能包含None）
        prompt_indices = self.get_prompt_indices(p)


        layer1_prompt = self.apply_prompts(
            layer1, self.prompt_layer1, prompt_weights, prompt_indices
        )


        layer2_prompt = self.apply_prompts(
            layer2, self.prompt_layer2, prompt_weights, prompt_indices
        )


        layer3_prompt = self.apply_prompts(
            layer3, self.prompt_layer3, prompt_weights, prompt_indices
        )

        return a, v,out,p, layer1_prompt, layer2_prompt, layer3_prompt

