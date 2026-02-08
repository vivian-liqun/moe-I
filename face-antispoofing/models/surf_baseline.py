import torch.nn as nn
import torch
import pathlib
import sys
if sys.platform == 'win32':
    # 临时保存原始的PosixPath
    temp = pathlib.PosixPath
    # 将PosixPath替换为WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath

from models.resnet18_se import resnet18_se

from lib.model_arch import modality_drop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class student_base(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,

                                         )

        self.mu_dul_backbone = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )
        self.b_backbone = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )
        self.r_backbone = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(model_resnet18_se_1.fc)

    def forward(self, img_rgb, img_ir, img_depth):

        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)


        x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)


        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)

        layer3 = self.shared_bone[0](x)

        layer4 = self.shared_bone[1](layer3)

        layer5 = self.mu_dul_backbone(layer4)
        b = self.b_backbone(layer5)
        r = self.r_backbone(layer5)
        layer6 = r * layer5 + b
        features = self.pooling(layer6)
        features = features.view(features.shape[0], -1)

        x = self.head(features)

        return x, layer3, layer4,layer6,p


def mask_modalities_with_zeros(x_rgb, x_ir, x_depth, p):
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
    p_depth = p[:, 2].view(B, 1, 1, 1).expand_as(x_depth)

    # 创建布尔掩码（True表示保留，False表示用0填充）[2,3](@ref)
    mask_rgb = p_rgb.bool()
    mask_ir = p_ir.bool()
    mask_depth = p_depth.bool()

    # 应用掩码：将需要掩码的位置填充为0[1,2](@ref)
    masked_rgb = x_rgb.masked_fill(~mask_rgb, 0.0)  # ~mask_rgb表示需要填充的位置
    masked_ir = x_ir.masked_fill(~mask_ir, 0.0)
    masked_depth = x_depth.masked_fill(~mask_depth, 0.0)

    return masked_rgb, masked_ir, masked_depth

class teacher_base_valid(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,

                                         )

        self.mu_dul_backbone = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )
        # self.mamba_layer_r = MambaLayer(dim=128)
        # self.mamba_layer_i = MambaLayer(dim=128)
        # self.mamba_layer_d = MambaLayer(dim=128)
        #
        # self.logvar_dul_backbone = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        # )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(model_resnet18_se_1.fc)

    def forward(self, img_rgb, img_ir, img_depth,p):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        x_rgb, x_ir, x_depth=mask_modalities_with_zeros(x_rgb, x_ir, x_depth,p)

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)


        layer3 = self.shared_bone[0](x)

        layer4 = self.shared_bone[1](layer3)

        layer5 = self.mu_dul_backbone(layer4)

        features = self.pooling(layer5)
        features = features.view(features.shape[0], -1)

        x = self.head(features)

        return x, layer3, layer4,layer5

class teacher_base(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,

                                         )

        self.mu_dul_backbone = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )
        # self.mamba_layer_r = MambaLayer(dim=128)
        # self.mamba_layer_i = MambaLayer(dim=128)
        # self.mamba_layer_d = MambaLayer(dim=128)
        #
        # self.logvar_dul_backbone = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        # )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(model_resnet18_se_1.fc)

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)



        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)

        layer3 = self.shared_bone[0](x)

        layer4 = self.shared_bone[1](layer3)

        layer5 = self.mu_dul_backbone(layer4)

        features = self.pooling(layer5)
        features = features.view(features.shape[0], -1)

        x = self.head(features)

        return x, layer3, layer4,layer5

def TeacherWithPrompt(teacher_instance):
    """
    装饰器函数：接收已经加载预训练参数的teacher_base实例
    返回一个增强的模型类，该模型会冻结原始参数并添加可训练的提示参数

    Args:
        teacher_instance: 已经加载预训练参数的teacher_base实例
    """

    class PromptEnhancedTeacher(nn.Module):
        def __init__(self):
            super().__init__()

            # 保存传入的教师模型实例
            self.teacher = teacher_instance

            # 冻结教师模型的所有参数
            for param in self.teacher.parameters():
                param.requires_grad = False

            # 初始化可训练的提示参数
            self._init_prompt_parameters()

        def _init_prompt_parameters(self):
            """初始化7种缺失类型对应的提示参数"""
            self.num_prompt_types = 7

            # 为layer3添加提示参数 [7, 256, 7, 7]
            self.prompt_layer3 = nn.Parameter(
                torch.randn(self.num_prompt_types, 256, 7, 7)*1
            )

            # 为layer4添加提示参数 [7, 512, 4, 4]
            self.prompt_layer4 = nn.Parameter(
                torch.randn(self.num_prompt_types, 512, 4, 4)*1
            )

            # 为features添加提示参数 [7, 512, 4, 4]
            self.prompt_features = nn.Parameter(
                torch.randn(self.num_prompt_types, 512, 4, 4)*1
            )
            self.missing_type_mapping = {
                (1, 0, 0): 0,
                (0, 1, 0): 1,
                (0, 0, 1): 2,
                (1, 1, 0): 3,
                (1, 0, 1): 4,
                (0, 1, 1): 5,
                (1, 1, 1): 6

            }

        def get_prompt_indices(self, p):
            """根据缺失情况p获取对应的提示索引，不在映射中的返回None"""
            batch_size = p.size(0)
            prompt_indices = []

            for i in range(batch_size):
                key = (
                    int(p[i, 0].item()),
                    int(p[i, 1].item()),
                    int(p[i, 2].item())
                )
                if key in self.missing_type_mapping:
                    prompt_indices.append(self.missing_type_mapping[key])
                else:
                    prompt_indices.append(None)

            return prompt_indices

        def forward(self, img_rgb, img_ir, img_depth, p, prompt_weights):
            """
            前向传播，根据缺失情况添加带权重的提示

            Args:
                img_rgb: RGB图像 [B, C, H, W]
                img_ir: IR图像 [B, C, H, W]
                img_depth: Depth图像 [B, C, H, W]
                p: 缺失情况 [B, 3]，每行表示三个模态的存在情况
                prompt_weights: 权重矩阵 [B, 7]，每个样本对应7种提示类型的权重
            """
            # 获取提示索引
            prompt_indices = self.get_prompt_indices(p)
            batch_size = img_rgb.size(0)

            if not isinstance(prompt_weights, torch.Tensor):
                prompt_weights = torch.tensor(prompt_weights, dtype=torch.float32, device=img_rgb.device)

                # 调整权重形状以便广播 [1, 7] -> [1, 7, 1, 1, 1]
            weights_reshaped = prompt_weights.view(1, 7, 1, 1, 1)

            # 使用教师模型计算基础特征（不计算梯度）
            with torch.no_grad():
                x_rgb = self.teacher.special_bone_rgb(img_rgb)
                x_ir = self.teacher.special_bone_ir(img_ir)
                x_depth = self.teacher.special_bone_depth(img_depth)

                x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
                layer3 = self.teacher.shared_bone[0](x)

                # 为需要提示的类型添加带权重的layer3提示
            layer3_with_prompt = layer3.clone()
            for i, idx in enumerate(prompt_indices):
                if idx is not None:
                        # 获取对应类型的权重 [1] -> [1, 1, 1, 1] 以便广播
                    weight = weights_reshaped[0, idx]  # 形状 [1, 1, 1]
                        # 提示参数乘以权重后添加到特征
                    weighted_prompt = self.prompt_layer3[idx] * weight
                    layer3_with_prompt[i] = layer3[i] + weighted_prompt

                # 继续前向传播
            with torch.no_grad():
                layer4 = self.teacher.shared_bone[1](layer3)

                # 为需要提示的类型添加带权重的layer4提示
            layer4_with_prompt = layer4.clone()
            for i, idx in enumerate(prompt_indices):
                if idx is not None:
                    weight = weights_reshaped[0, idx]  # 形状 [1, 1, 1]
                    weighted_prompt = self.prompt_layer4[idx] * weight
                    layer4_with_prompt[i] = layer4[i] + weighted_prompt

                # 继续前向传播
            with torch.no_grad():
                features_before_pooling = self.teacher.mu_dul_backbone(layer4)

                # 为需要提示的类型添加带权重的features提示
            features_with_prompt = features_before_pooling.clone()
            for i, idx in enumerate(prompt_indices):
                if idx is not None:
                    weight = weights_reshaped[0, idx]  # 形状 [1, 1, 1]
                    weighted_prompt = self.prompt_features[idx] * weight
                    features_with_prompt[i] = features_before_pooling[i] + weighted_prompt

            # 池化和全连接层
            with torch.no_grad():
                features_pooled = self.teacher.pooling(features_before_pooling)
                features_flattened = features_pooled.view(features_pooled.shape[0], -1)
                output = self.teacher.head(features_flattened)

            return output, layer3_with_prompt, layer4_with_prompt, features_with_prompt


    return PromptEnhancedTeacher



