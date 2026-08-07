import re
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from sentence_transformers import SentenceTransformer 

# ==========================================
# 1. 认知测量词典 (The Cognitive Lexicon)
# ==========================================
class CognitiveDictionary:
    def __init__(self):
        self.dict_S = {"then", "and", "so", "because", "first", "next", "also"}
        self.dict_M = {"think", "guess", "maybe", "wait", "wrong", "sure", "check", "know"}
        self.dict_A = {"frustrating", "ugh", "hard", "stupid", "yay", "mad", "suck"}
        self.dict_T = {"rotate", "triangle", "edge", "fit", "turn", "long", "shape", "size", "match", "align"}

    def detect_task_presence(self, text_frame):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text_frame.lower())
        words = set(clean_text.split())
        if self.dict_T.intersection(words):
            return 1.0 
        else:
            return 0.0

# ==========================================
# 2. 流式雷达探头 (The Embedder)
# ==========================================
class TAPStreamEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2', window_size=12, stride=1):
        self.sbert = SentenceTransformer(model_name)
        self.window_size = window_size
        self.stride = stride

    def extract(self, raw_text):
        words = raw_text.replace('\n', ' ').split()
        total_words = len(words)
        
        frames = []
        for i in range(0, max(1, total_words - self.window_size + 1), self.stride):
            frames.append(" ".join(words[i : i + self.window_size]))
            
        with torch.no_grad():
            h_t_sequence = self.sbert.encode(frames, convert_to_tensor=True)
            
        return frames, h_t_sequence

# ==========================================
# 3. 多头门控裁判 (The Gating Network)
# ==========================================
class TAAGGatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 4)

    def forward(self, h_t):
        s_t = self.projection(h_t)
        g_t = F.softmax(s_t, dim=-1)
        return g_t

# ==========================================
# 4. 引力计算器 (The Gravity Segmenter)
# ==========================================
class GravitySegmenter:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.omega = torch.tensor([0.1, 0.8, 1.2]) 
        self.beta = 0.5 

    def calculate_gravity(self, g_t_weights, has_task_word):
        g_S = g_t_weights[:, 0]
        g_M = g_t_weights[:, 1]
        g_A = g_t_weights[:, 2]
        g_T = g_t_weights[:, 3] 
        
        task_enhancement = 1.0 + self.beta * has_task_word
        pull = g_T * task_enhancement
        friction = (g_S * self.omega[0]) + (g_M * self.omega[1]) + (g_A * self.omega[2])
        G_t = pull - friction
        
        return G_t.tolist()

    # 包含“非对称相位修正”的终极切割逻辑
    def segment(self, words, G_t_sequence, stride, window_size):
        edus = []
        current_start_idx = 0
        was_in_valley = G_t_sequence[0] < self.threshold
        
        for i, g_value in enumerate(G_t_sequence):
            is_in_valley = g_value < self.threshold
            
            if is_in_valley != was_in_valley:
                if is_in_valley:
                    # 跌入谷底，铡刀向后偏移，保留无辜的动作词
                    cut_point = i * stride + window_size - 1
                else:
                    # 爬出谷底，铡刀原位落下
                    cut_point = i * stride
                
                cut_point = min(cut_point, len(words))
                
                if cut_point > current_start_idx:
                    edu_text = " ".join(words[current_start_idx : cut_point])
                    edus.append(edu_text)
                    current_start_idx = cut_point
            
            was_in_valley = is_in_valley
                
        if current_start_idx < len(words):
            edus.append(" ".join(words[current_start_idx:]))
            
        return edus

# ==========================================
# 5. 终极实战引擎 (Execution)
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 PsyMAS 维度感知引力引擎 (模拟已训练状态)...\n")
    
    raw_tap = "I rotate the shape. I align the triangle. Wait, I think it is wrong. It is frustrating. Let me turn this edge."
    
    cog_dict = CognitiveDictionary() 
    # 短句演示，设置 window_size=3
    embedder = TAPStreamEmbedder(window_size=3, stride=1) 
    segmenter = GravitySegmenter(threshold=0.4) 
    
    print("1. 提取高维语义特征...")
    frames, _ = embedder.extract(raw_tap)
    
    print("2. 模拟已训练的神经网络，分配注意力预算...")
    simulated_g_t = []
    for frame in frames:
        clean_words = set(re.sub(r'[^a-zA-Z\s]', '', frame.lower()).split())
        
        if cog_dict.dict_A.intersection(clean_words):
            simulated_g_t.append([0.1, 0.1, 0.8, 0.0])
        elif cog_dict.dict_M.intersection(clean_words):
            simulated_g_t.append([0.1, 0.8, 0.1, 0.0])
        elif cog_dict.dict_T.intersection(clean_words):
            simulated_g_t.append([0.1, 0.0, 0.1, 0.8])
        else:
            simulated_g_t.append([0.8, 0.1, 0.0, 0.1])
            
    g_t_seq = torch.tensor(simulated_g_t)
    
    print("3. 利用认知词典探测空间动作，计算引力增强项...")
    has_task_word_list = [cog_dict.detect_task_presence(frame) for frame in frames]
    has_task_word_tensor = torch.tensor(has_task_word_list)
    
    print("4. 计算认知引力，寻找坍塌断点...")
    G_t_seq = segmenter.calculate_gravity(g_t_seq, has_task_word_tensor)
    
    words_list = raw_tap.replace('\n', ' ').split()
    
    # 🌟 修复点：在这里把 embedder.window_size 传进去
    final_edus = segmenter.segment(words_list, G_t_seq, embedder.stride, embedder.window_size)
    
    print("\n✅ 自动化认知法医切割完毕！提取出的 EDU 证据块如下：")
    for idx, edu in enumerate(final_edus):
        print(f"EDU {idx + 1}: {edu}")