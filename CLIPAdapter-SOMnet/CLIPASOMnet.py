import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------
# 核心模块定义
# --------------------------

class ShortcutAdapter(nn.Module):
    '''
    """带残差连接的适配器结构"""
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        hidden_dim = input_dim // reduction
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.layers(x.float()) + self.shortcut(x.float())
    '''
    def __init__(self, input_dim, reduction=8):
        super().__init__()
        hidden_dim = input_dim // reduction
        '''
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),          # 输入维度BN
            nn.ReLU(inplace=True),              # 激活函数
            nn.Linear(input_dim, hidden_dim),   # 降维线性层
            nn.BatchNorm1d(hidden_dim),         # 隐藏层BN
            nn.ReLU(inplace=True),              # 激活函数
            #nn.Linear(hidden_dim, hidden_dim),     # 恢复维度线性层
            #nn.BatchNorm1d(hidden_dim),         # 隐藏层BN
            #nn.ReLU(inplace=True),              # 激活函数
            nn.Linear(hidden_dim, input_dim)     # 恢复维度线性层
        )
        '''
      
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 降维线性层
            nn.BatchNorm1d(hidden_dim),          # 输入维度BN
            nn.ReLU(inplace=True),              # 激活函数
            nn.Linear(hidden_dim, hidden_dim),   # 降维线性层
            nn.BatchNorm1d(hidden_dim),         # 隐藏层BN
            nn.ReLU(inplace=True),              # 激活函数
            nn.Linear(hidden_dim, input_dim)     # 恢复维度线性层
        )
        
        self.shortcut = nn.Linear(input_dim, input_dim) #nn.Identity()           # 残差连接捷径

    def forward(self, x):
        # 保持数据类型一致性并执行残差相加
        return self.layers(x.float()) + self.shortcut(x.float())#self.shortcut(x.float())+
    def freeze_shortcut(self):
        """冻结shortcut参数，只训练layers参数"""
        # 冻结shortcut
        for param in self.shortcut.parameters():
            param.requires_grad = False
        # 确保layers可训练
        for param in self.layers.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
# --------------------------
# 第一阶段：预训练适配器
# --------------------------

def pretrain_adapters(visual_adapter, text_adapter, clip_model, train_loader, device, num_epochs=10):
    # 冻结shortcut参数，只训练layers参数
    visual_adapter.freeze_shortcut()
    text_adapter.freeze_shortcut()
    # 初始化适配器
    visual_dim = clip_model.visual.output_dim
    text_dim = clip_model.text_projection.shape[-1]
    #visual_adapter = ShortcutAdapter(visual_dim).to(device)
    #text_adapter = ShortcutAdapter(text_dim).to(device)
    
    # 预编码所有文本特征
    class_names = train_loader.dataset.classes
    text_descriptions = [f"a photo of a {name}" for name in class_names]
    text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).float()

    # 优化器 - 只优化需要梯度的参数
    visual_trainable_params = [p for p in visual_adapter.parameters() if p.requires_grad]
    text_trainable_params = [p for p in text_adapter.parameters() if p.requires_grad]
    
    # 优化器
    optimizer = torch.optim.Adam(
         visual_trainable_params + text_trainable_params,
        lr=1e-3
    )
    
    # 预训练循环
    for epoch in range(num_epochs):
        visual_adapter.train()
        text_adapter.train()
        total_loss = 0
        
        for images, _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}"):
            images = images.to(device)
            
            # 提取图像特征
            with torch.no_grad():
                img_feat = clip_model.encode_image(images).float()
            
            # 计算适配器输出
            visual_out = visual_adapter(img_feat)
            text_out = text_adapter(text_features)
            
            # 损失函数：L2范数最小化
            loss_visual = torch.mean(torch.norm(visual_out, dim=1))
            loss_text = torch.mean(torch.norm(text_out, dim=1))
            loss = loss_visual + loss_text
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Pretrain Loss: {total_loss/len(train_loader):.4f}")
        
    return visual_adapter.state_dict(), text_adapter.state_dict()

# --------------------------
# 第二阶段：监督微调
# --------------------------

class FinetuneModel(nn.Module):
    def __init__(self, clip_model, visual_adapter, text_adapter, alpha=0.5):
        super().__init__()
        self.clip = clip_model.float()
        for param in self.clip.parameters():
            param.requires_grad_(False)
        
        # 加载预训练适配器
        self.visual_adapter = visual_adapter
        self.text_adapter = text_adapter
        self.alpha = alpha
        # 解冻所有适配器参数用于微调
        self.visual_adapter.unfreeze_all()
        self.text_adapter.unfreeze_all()

    def forward(self, images, text_features):
        # 提取图像特征
        with torch.no_grad():
            img_feat = self.clip.encode_image(images).float()
        
        # 特征融合
        #img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        visual_out =  self.visual_adapter(img_feat) +  img_feat
        text_out = self.text_adapter(text_features) + text_features
        
        # 归一化
        visual_out = visual_out / visual_out.norm(dim=-1, keepdim=True)
        text_out = text_out / text_out.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.clip.logit_scale.exp().float()
        return logit_scale * (visual_out @ text_out.T)

def finetune(clip_model, visual_weights, text_weights, train_loader, test_loader, device):
    # 初始化模型
    visual_adapter = ShortcutAdapter(clip_model.visual.output_dim)
    text_adapter = ShortcutAdapter(clip_model.text_projection.shape[-1])
    visual_adapter.load_state_dict(visual_weights)
    text_adapter.load_state_dict(text_weights)
    
    model = FinetuneModel(clip_model, visual_adapter, text_adapter).to(device)
    
    # 预编码文本特征
    class_names = train_loader.dataset.classes
    text_descriptions = [f"a photo of a {name}" for name in class_names]
    text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).float()
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': model.visual_adapter.parameters()},
        {'params': model.text_adapter.parameters()}
    ], lr=1e-4, weight_decay=1e-4)
    
    # 训练循环
    best_acc = 0
    for epoch in range(15):
        model.train()
        total_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images, text_features)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 测试
        acc = test(model, test_loader, text_features, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_ft_model.pth")
    
    print(f"Best Fine-tuned Accuracy: {best_acc:.2%}")
    return visual_adapter.state_dict(), text_adapter.state_dict()

def test(model, test_loader, text_features, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, text_features)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --------------------------
# 主流程
# --------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                          (0.26862954, 0.26130258, 0.27577711))
    ])
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
    visual_dim = clip_model.visual.output_dim
    text_dim = clip_model.text_projection.shape[-1]
    visual_adapter = ShortcutAdapter(visual_dim).to(device)
    text_adapter = ShortcutAdapter(text_dim).to(device)
    # 第一阶段：预训练适配器
    
    print("=== Stage 1: Pretraining Adapters ===")
    visual_weights, text_weights = pretrain_adapters(visual_adapter, text_adapter, 
        clip_model, train_loader, device, num_epochs=20
    )
    
    # 第二阶段：监督微调
    print("\n=== Stage 2: Supervised Finetuning ===")
    visual_weights, text_weights=finetune(clip_model, visual_weights, text_weights, train_loader, test_loader, device)
    
    visual_adapter.load_state_dict(visual_weights)
    text_adapter.load_state_dict(text_weights)
    # 第一阶段：预训练适配器
    print("=== Stage 1: Pretraining Adapters ===")
    visual_weights, text_weights = pretrain_adapters(visual_adapter, text_adapter, 
        clip_model, train_loader, device, num_epochs=20
    )
    
    # 第二阶段：监督微调
    print("\n=== Stage 2: Supervised Finetuning ===")
    visual_weights, text_weights=finetune(clip_model, visual_weights, text_weights, train_loader, test_loader, device)
    

if __name__ == "__main__":
    main()