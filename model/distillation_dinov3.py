import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import init_weights
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel

class DinoLVSMComparator(nn.Module):

    def __init__(
        self, 
        lvsm_feature_dim, 
        dino_feature_dim, 
        num_layers=2,
        dino_model_type="vitl16"
        ):
        super().__init__()
        model_name = self.SUPPORTED_MODELS[dino_model_type]
        self.dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self.dino_model = AutoModel.from_pretrained(model_name)
        
        self.dino_model.eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        proj_layers = []
        for i in range(num_layers):
            if i == 0:
                proj_layers.append(nn.Linear(lvsm_feature_dim, dino_feature_dim))
            else:
                proj_layers.append(nn.Linear(dino_feature_dim, dino_feature_dim))
            proj_layers.append(nn.SiLU())
        self.proj_head = nn.Sequential(*proj_layers)
        self.proj_head.apply(init_weights)
        
        self.l2_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    @staticmethod
    def mean_flat(x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))
        
    def get_dino_patch_features(self, images, layer_idx=-2):
        """提取DINOv3模型指定中间层的patch特征（不含[CLS] token）"""
        B, T, H, W = images.shape
        
        images_flat = rearrange(images, "b t h w -> (b t) h w") 
        inputs = self.dino_processor(images_flat, return_tensors="pt")
        
        device = next(self.dino_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dino_model(** inputs, output_hidden_states=True)
            target_hidden = outputs.hidden_states[layer_idx]
            patch_features = target_hidden[:, 1:, :]
            patch_features = rearrange(patch_features, "(b t) n d -> b t n d", b=B, t=T)
        
        return patch_features
    
    def project_lvsm_features(self, lvsm_mid_features):
        return self.proj_head(lvsm_mid_features)
    
    def compute_loss(self, lvsm_features, dino_features, loss_type='l2'):
        """
        计算LVSM投影特征与DINOv3特征之间的损失
        
        Args:
            lvsm_features: 投影后的LVSM特征，形状为[B, T, N_lvsm, D]
            dino_features: DINOv3提取的特征，形状为[B, T, N_dino, D]
            loss_type: 损失函数类型，可选'l2', 'cos_sim', 'smooth_l1', 'cross_entropy'
            
        Returns:
            计算得到的损失值
        """
        assert lvsm_features.shape[:2] == dino_features.shape[:2], \
            f"批次和时间维度不匹配：LVSM为{lvsm_features.shape[:2]}，DINO为{dino_features.shape[:2]}"
        assert lvsm_features.shape[-1] == dino_features.shape[-1], \
            f"特征维度不匹配：LVSM为{lvsm_features.shape[-1]}，DINO为{dino_features.shape[-1]}"
        
        lvsm_flat = rearrange(lvsm_features, "b t n d -> (b t) n d")
        dino_flat = rearrange(dino_features, "b t n d -> (b t) n d")
        
        if loss_type == 'l2':
            return self.mean_flat(self.l2_loss(lvsm_flat, dino_flat))
            
        elif loss_type == 'cos_sim':
            lvsm_norm = F.normalize(lvsm_flat, p=2, dim=-1)
            dino_norm = F.normalize(dino_flat, p=2, dim=-1)
            
            cos_sim = self.mean_flat(torch.sum(lvsm_norm * dino_norm, dim=-1))
            return 1 - cos_sim
            
        elif loss_type == 'smooth_l1':
            return self.mean_flat(self.smooth_l1_loss(lvsm_flat, dino_flat))
            
        elif loss_type == 'cross_entropy':
            _, n_patches_dino, _ = dino_flat.shape

            lvsm_norm = F.normalize(lvsm_flat, p=2, dim=-1)
            dino_norm = F.normalize(dino_flat, p=2, dim=-1)
            sim_matrix = torch.matmul(lvsm_norm, dino_norm.transpose(1, 2))
            labels = sim_matrix.argmax(dim=2)

            loss = F.cross_entropy(
                sim_matrix.reshape(-1, n_patches_dino),
                labels.reshape(-1)
            )
            return loss
            
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}，可选类型为'l2', 'cos_sim', 'smooth_l1', 'cross_entropy'")
    
    def distillation_block(self, images_clean, lvsm_mid_features, layer_idx=-2, loss_type='l2'):
        """
        蒸馏过程封装：接收干净图片集和LVSM中间层信息，输出最终蒸馏损失
        
        Args:
            images_clean: 干净图片集，形状为[B, T, H, W]
            lvsm_mid_features: LVSM模型的中间层特征，形状为[B, T, N_lvsm, D_lvsm]
            layer_idx: DINOv3提取特征的中间层索引，默认-2
            loss_type: 损失函数类型，默认'l2'，可选'l2', 'cos_sim', 'smooth_l1', 'cross_entropy'
            
        Returns:
            distill_loss: 蒸馏损失，可直接用于反向传播更新LVSM参数
            optional_outputs: 字典，包含中间特征（教师特征、投影后学生特征），便于调试分析
        """
        device = next(self.dino_model.parameters()).device
        images_clean = images_clean.to(device)
        lvsm_mid_features = lvsm_mid_features.to(device)
        
        dino_teacher_feats = self.get_dino_patch_features(images_clean, layer_idx)
        lvsm_student_proj = self.project_lvsm_features(lvsm_mid_features)
        
        distill_loss = self.compute_loss(
            lvsm_features=lvsm_student_proj,
            dino_features=dino_teacher_feats,
            loss_type=loss_type
        )
        
        optional_outputs = {
            "dino_teacher_features": dino_teacher_feats,
            "lvsm_projected_features": lvsm_student_proj
        }
        
        return distill_loss, optional_outputs

