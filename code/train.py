import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 认知测量词典 (带有打标功能 assign_label)
# ==========================================
class CognitiveDictionary:
    def __init__(self):
        self.dict_S = {"then", "and", "so", "because", "first", "next", "also"}
        self.dict_M = {"think", "guess", "maybe", "wait", "wrong", "sure", "check", "know"}
        #self.dict_A = {"frustrating", "ugh", "hard", "stupid", "yay", "mad", "suck", "confused"}
        self.dict_A = {"confused", "confusing", "tricky", "easy", "simple", "good", "bad", "laughter"}
        #self.dict_T = {"rotate", "triangle", "edge", "fit", "turn", "long", "shape", "size", "match", "align", "box"}
        # 空间任务流 (Task): 基于 Cannon et al. (2007) 框架与真实频次重构
        self.dict_T = {
            # 1. 形状与特征 (Shape & Features of an object): 
            # 捕获了高频的具体几何名词和部件名词
            "triangle", "triangles", "square", "squares", "rectangle", "shape", "shapes", 
            "circle", "piece", "pieces", "line", "lines", "angle", "point", "corner", "edge",
            
            # 2. 尺寸与形态特征 (Size or shape):
            # 吸收了口语中高频出现的比较级和具象形容词
            "big", "bigger", "small", "smaller", "little", "long", "longer", "short", 
            "wide", "wider", "skinny", "skinnier", "straight", "flat", "curve", "size", "sizes",
            
            # 3. 相对位置与方向 (Relative location & Orientation):
            # 剔除了极易引起歧义的介词(up/down/in/out)，仅保留具有明确空间指向的实体方位词
            "top", "bottom", "side", "sides", "middle", "center", "left", "right", "front", "back",
            
            # 4. 空间操作与变换 (Rotation & Relative distance):
            # 结合了 Cannon 的理论动作词与学生常用的平替动作词
            "rotate", "turn", "turned", "flip", "flipped", "fit", "fits", "match", "matches", 
            "put", "make", "makes", "same", "equal", "together", "apart"
        }
        # 标签映射：S=0, M=1, A=2, T=3
        self.label_map = {'S': 0, 'M': 1, 'A': 2, 'T': 3}

    def assign_label(self, text_frame):
        """执行层级否决制打标，生成银标"""
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text_frame.lower())
        words = set(clean_text.split())
        
        if self.dict_A.intersection(words): return self.label_map['A']
        if self.dict_M.intersection(words): return self.label_map['M']
        if self.dict_T.intersection(words): return self.label_map['T']
        if self.dict_S.intersection(words): return self.label_map['S']
        
        return self.label_map['S']


# ==========================================
# 2. 流式雷达探头 (特征压缩)
# ==========================================
class TAPStreamEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2', window_size=8, stride=1):
        self.sbert = SentenceTransformer(model_name)
        self.window_size = window_size
        self.stride = stride

    def extract(self, raw_text):
        if pd.isna(raw_text) or not isinstance(raw_text, str):
            return [], None
            
        words = raw_text.replace('\n', ' ').split()
        total_words = len(words)
        
        frames = []
        for i in range(0, max(1, total_words - self.window_size + 1), self.stride):
            frames.append(" ".join(words[i : i + self.window_size]))
            
        if not frames:
            return [], None
            
        with torch.no_grad():
            h_t_sequence = self.sbert.encode(frames, convert_to_tensor=True)
            
        return frames, h_t_sequence

# ==========================================
# 3. 多头门控裁判 (我们要训练的神经网络)
# ==========================================
class TAAGGatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 4)

    def forward(self, h_t):
        s_t = self.projection(h_t)
        return s_t

# ==========================================
# 4. 数据冶炼厂：(带有宽容解码机制)
# ==========================================
def build_training_dataset(csv_path, embedder, labeler):
    print(f"📂 正在读取数据集: {csv_path}...")
    
    # 强力容错读取机制
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='windows-1252')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    
    item_columns = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
    all_vectors = []
    all_labels = []
    
    for index, row in df.iterrows():
        for col in item_columns:
            if col in df.columns:
                text = row[col]
                if pd.notna(text):
                    text = str(text).replace('Interviewee:', '').strip()
                    if len(text) < 5: continue
                    
                    frames, vectors = embedder.extract(text)
                    if vectors is not None:
                        labels = [labeler.assign_label(f) for f in frames]
                        all_vectors.append(vectors)
                        all_labels.extend(labels)
                        
    X_train = torch.cat(all_vectors, dim=0)
    y_train = torch.tensor(all_labels, dtype=torch.long)
    
    return X_train, y_train

# ==========================================
# 5. 引擎启动：训练与持久化
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 PsyMAS 预训练工厂...\n")
    
    # 初始化组件
    cog_dict = CognitiveDictionary()
    embedder = TAPStreamEmbedder(window_size=8, stride=1)
    gating_network = TAAGGatingNetwork()
    
    # 请确保路径与你报错时的路径一致
    csv_file = "../data/ThinkAloudOA.csv" 
    
    if not os.path.exists(csv_file):
        print(f"❌ 未找到文件: {csv_file}")
    else:
        # 提取数据
        X_train, y_train = build_training_dataset(csv_file, embedder, cog_dict)
        
        print(f"\n📊 数据集构建完成！")
        print(f"- 提取到的总视窗数量: {len(X_train)}")
        print(f"- 银标分布 -> S:{sum(y_train==0)}, M:{sum(y_train==1)}, A:{sum(y_train==2)}, T:{sum(y_train==3)}")
        
        # 封装为 DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 配置优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(gating_network.parameters(), lr=0.005)
        epochs = 30 # 设置 30 轮快速验证
        
        print("\n🔥 开始微调门控网络权重...")
        gating_network.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                logits = gating_network(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(dataloader)
                accuracy = (correct / total) * 100
                print(f"Epoch {epoch+1:02d}/{epochs} | 联合损失: {avg_loss:.4f} | 字典拟合度: {accuracy:.1f}%")

        print("\n✅ 训练完成！门控网络已内化静态字典的认知规律。")
        
        save_path = "psymas_gating_weights.pth"
        torch.save(gating_network.state_dict(), save_path)
        print(f"💾 模型权重已成功保存至: {os.path.abspath(save_path)}")