import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 核心推理组件 (The Inference Engine)
# ==========================================
class TAPStreamEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2', window_size=5, stride=1):
        self.sbert = SentenceTransformer(model_name)
        self.window_size = window_size
        self.stride = stride

    def extract(self, raw_text):
        if not isinstance(raw_text, str) or not raw_text.strip():
            return [], None
        words = raw_text.replace('\n', ' ').split()
        if len(words) < self.window_size:
            return [], None
        
        frames = [" ".join(words[i : i + self.window_size]) for i in range(0, max(1, len(words) - self.window_size + 1), self.stride)]
        with torch.no_grad():
            h_t_sequence = self.sbert.encode(frames, convert_to_tensor=True)
        return frames, h_t_sequence

class TAAGGatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 4)

    def forward(self, h_t):
        s_t = self.projection(h_t)
        return F.softmax(s_t, dim=-1)

class CognitiveDictionary:
    def __init__(self):
        self.dict_T = {
            "triangle", "triangles", "square", "squares", "rectangle", "shape", "shapes", 
            "circle", "piece", "pieces", "line", "lines", "angle", "point", "corner", "edge",
            "big", "bigger", "small", "smaller", "little", "long", "longer", "short", 
            "wide", "wider", "skinny", "skinnier", "straight", "flat", "curve", "size", "sizes",
            "top", "bottom", "side", "sides", "middle", "center", "left", "right", "front", "back",
            "rotate", "turn", "turned", "flip", "flipped", "fit", "fits", "match", "matches", 
            "put", "make", "makes", "same", "equal", "together", "apart"
        }

    def detect_task_presence(self, text_frame):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text_frame.lower())
        return 1.0 if self.dict_T.intersection(set(clean_text.split())) else 0.0

class GravitySegmenter:
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.omega = torch.tensor([0.1, 0.8, 1.2]) 
        self.beta = 0.5 

    def calculate_gravity(self, g_t_weights, has_task_word):
        pull = g_t_weights[:, 3] * (1.0 + self.beta * has_task_word)
        friction = (g_t_weights[:, 0] * self.omega[0]) + (g_t_weights[:, 1] * self.omega[1]) + (g_t_weights[:, 2] * self.omega[2])
        return (pull - friction).tolist()

    def segment(self, words, G_t_sequence, stride, window_size):
        edus = []
        current_start_idx = 0
        was_in_valley = G_t_sequence[0] < self.threshold
        for i, g_value in enumerate(G_t_sequence):
            is_in_valley = g_value < self.threshold
            if is_in_valley != was_in_valley:
                cut_point = i * stride + window_size - 1 if is_in_valley else i * stride
                cut_point = min(cut_point, len(words))
                if cut_point > current_start_idx:
                    edus.append(" ".join(words[current_start_idx : cut_point]))
                    current_start_idx = cut_point
            was_in_valley = is_in_valley
        if current_start_idx < len(words):
            edus.append(" ".join(words[current_start_idx:]))
        return edus

class MJDinaAdapter:
    def __init__(self, embedder, gating_network):
        self.embedder = embedder
        self.gating_network = gating_network
        self.labels = ["S", "M", "A", "T"]

    def vectorize_edu(self, edu_text):
        frames, h_t_seq = self.embedder.extract(edu_text)
        if h_t_seq is None: 
            return [0,0,0,0], [0.0, 0.0, 0.0, 0.0], "None"
            
        with torch.no_grad():
            g_t_weights = self.gating_network(h_t_seq)
            
        avg_weights = torch.mean(g_t_weights, dim=0)
        dominant_dim = torch.argmax(avg_weights).item()
        
        q_vector = [0, 0, 0, 0]
        q_vector[dominant_dim] = 1
        
        return q_vector, avg_weights.tolist(), self.labels[dominant_dim]

# ==========================================
# 2. 批量切割流水线 (Batch Pipeline)
# ==========================================
def run_batch_segmentation(input_csv, output_csv, weights_path):
    print(f"🚀 初始化 PsyMAS 批量法医引擎...")
    
    # 1. 检查必要文件
    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到输入数据文件 '{input_csv}'")
        return
    if not os.path.exists(weights_path):
        print(f"❌ 错误: 找不到模型权重 '{weights_path}'")
        return

    # 2. 装载组件与大脑
    embedder = TAPStreamEmbedder(window_size=5, stride=1)
    gating_network = TAAGGatingNetwork()
    gating_network.load_state_dict(torch.load(weights_path))
    gating_network.eval()
    
    cog_dict = CognitiveDictionary()
    segmenter = GravitySegmenter(threshold=0.35)
    adapter = MJDinaAdapter(embedder, gating_network)
    
    print(f"✅ 模型与组件装载完毕。正在读取: {input_csv}\n")
    df = pd.read_csv(input_csv)
    
    output_data = []
    
    # 3. 遍历每个学生的发言
    for index, row in df.iterrows():
        student_id = row['ID']
        raw_response = str(row['Interviewee_Response'])
        
        # 将上一阶段用于分隔的 " | " 替换为空格，恢复为连续的意识流
        clean_response = raw_response.replace(" | ", " ")
        
        frames, h_t_seq = embedder.extract(clean_response)
        
        if h_t_seq is None:
            print(f"⚠️ 跳过 {student_id}: 文本过短。")
            continue
            
        # 执行引力计算与切分
        with torch.no_grad():
            g_t_seq = gating_network(h_t_seq)
            
        has_task_word_tensor = torch.tensor([cog_dict.detect_task_presence(f) for f in frames])
        G_t_seq = segmenter.calculate_gravity(g_t_seq, has_task_word_tensor)
        
        words_list = clean_response.split()
        final_edus = segmenter.segment(words_list, G_t_seq, embedder.stride, embedder.window_size)
        
        valid_edu_count = 0
        
        # 4. 对切出来的每个 EDU 进行多维特征矩阵化
        for edu in final_edus:
            # 丢弃过短的噪音碎片（少于3个词的嘟囔）
            if len(edu.split()) < 3:
                continue
                
            valid_edu_count += 1
            q_vec, w_vec, dom_label = adapter.vectorize_edu(edu)
            
            output_data.append({
                "Student_ID": student_id,
                "EDU_Index": valid_edu_count,
                "EDU_Text": edu,
                "Dominant_Dim": dom_label,
                "Q_S": q_vec[0], "Q_M": q_vec[1], "Q_A": q_vec[2], "Q_T": q_vec[3],
                "Weight_S": round(w_vec[0], 4),
                "Weight_M": round(w_vec[1], 4),
                "Weight_A": round(w_vec[2], 4),
                "Weight_T": round(w_vec[3], 4)
            })
            
        print(f"✓ {student_id} 处理完毕: 提取出 {valid_edu_count} 个有效 EDU。")

    # 5. 导出为 Q-矩阵结构表
    if output_data:
        out_df = pd.DataFrame(output_data)
        out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n🎉 批量切分大功告成！共生成 {len(output_data)} 条特征记录。")
        print(f"💾 Q-矩阵特征表已保存至: {os.path.abspath(output_csv)}")
    else:
        print("\n⚠️ 未提取到任何有效 EDU。")

# ==========================================
# 执行区
# ==========================================
if __name__ == "__main__":
    # 输入与输出的配置
    INPUT_FILE = "../data/ThinkAloudOA.csv"  # 上一步从 doc 提取的 CSV
    OUTPUT_FILE = "../data/Segmented_EDUs_QMatrix.csv"  # 最终输出的 Q-矩阵表
    WEIGHTS_FILE = "psymas_gating_weights.pth"         # 你训练出来的数字大脑
    
    run_batch_segmentation(INPUT_FILE, OUTPUT_FILE, WEIGHTS_FILE)