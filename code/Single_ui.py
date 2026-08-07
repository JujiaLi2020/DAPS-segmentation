import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 核心推理组件 (The Inference Engine)
# ==========================================
class TAPStreamEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2', window_size=8, stride=1):
        self.sbert = SentenceTransformer(model_name)
        self.window_size = window_size
        self.stride = stride

    def extract(self, raw_text):
        words = raw_text.replace('\n', ' ').split()
        if not words: return [], None
        total_words = len(words)
        frames = [" ".join(words[i : i + self.window_size]) for i in range(0, max(1, total_words - self.window_size + 1), self.stride)]
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
        # 挂载我们经过真实语料和 Cannon 理论校准的 T 词典
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
    def __init__(self, threshold=0.4):
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

    def vectorize_edu(self, edu_text):
        _, h_t_seq = self.embedder.extract(edu_text)
        if h_t_seq is None: return [0,0,0,0]
        with torch.no_grad():
            g_t_weights = self.gating_network(h_t_seq)
        avg_weights = torch.mean(g_t_weights, dim=0)
        dominant_dim = torch.argmax(avg_weights).item()
        q_vector = [0, 0, 0, 0]
        q_vector[dominant_dim] = 1
        return q_vector

# ==========================================
# 2. 系统初始化与模型挂载
# ==========================================
# 全局加载，避免每次点击按钮都重新加载模型
print("System initializing... Loading SBERT and PsyMAS weights.")
embedder = TAPStreamEmbedder(window_size=5, stride=1)
gating_network = TAAGGatingNetwork()

# 强行加载已训练的权重
try:
    gating_network.load_state_dict(torch.load("psymas_gating_weights.pth"))
    gating_network.eval()
    print("PsyMAS weights loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load weights. Using random initialization. Error: {e}")

cog_dict = CognitiveDictionary()
segmenter = GravitySegmenter(threshold=0.35)
adapter = MJDinaAdapter(embedder, gating_network)

# ==========================================
# 3. UI 交互逻辑 (Gradio Backend)
# ==========================================
def process_transcript(raw_text):
    """接收前端文本，运行切分，返回 HTML 格式的诊断报告"""
    if not raw_text.strip():
        return "<p style='color:red;'>Please enter a valid transcript.</p>"
        
    frames, h_t_seq = embedder.extract(raw_text)
    if h_t_seq is None:
        return "<p>Text too short to process.</p>"

    with torch.no_grad():
        g_t_seq = gating_network(h_t_seq)
        
    has_task_word_tensor = torch.tensor([cog_dict.detect_task_presence(f) for f in frames])
    G_t_seq = segmenter.calculate_gravity(g_t_seq, has_task_word_tensor)
    
    words_list = raw_text.replace('\n', ' ').split()
    final_edus = segmenter.segment(words_list, G_t_seq, embedder.stride, embedder.window_size)

    # 构建 HTML 输出界面
    dim_labels = ["S (Structure)", "M (Metacognition)", "A (Affect)", "T (Task)"]
    color_map = {0: "#e5e7eb", 1: "#fef08a", 2: "#fecaca", 3: "#bbf7d0"} # 灰, 黄, 红, 绿

    html_output = "<div style='font-family: sans-serif;'>"
    html_output += "<h3>Diagnostic Segmentation & Q-Matrix Vectors</h3>"
    
    for i, edu in enumerate(final_edus):
        if len(edu.split()) < 3: continue # 忽略过短的碎片
        
        q_vector = adapter.vectorize_edu(edu)
        dominant_idx = q_vector.index(1)
        bg_color = color_map[dominant_idx]
        dim_name = dim_labels[dominant_idx]
        
        html_output += f"""
        <div style='background-color: {bg_color}; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #d1d5db;'>
            <div style='font-weight: bold; margin-bottom: 4px; color: #374151;'>EDU {i+1} | Dominant: {dim_name} | Q-Vector: {q_vector}</div>
            <div style='font-size: 1.1em; color: #111827;'>"{edu}"</div>
        </div>
        """
    html_output += "</div>"
    return html_output

# ==========================================
# 4. 构建前端界面 (Gradio UI)
# ==========================================
demo = gr.Interface(
    fn=process_transcript,
    inputs=gr.Textbox(
        lines=8, 
        placeholder="Paste the student's Think-Aloud Protocol here... e.g., 'I will put the skinny piece on top. Wait, it looks wrong. Let me turn it around.'",
        label="Raw Think-Aloud Transcript"
    ),
    outputs=gr.HTML(label="PsyMAS Analysis Output"),
    title="PsyMAS: Cognitive Forensics Engine",
    description="This tool applies the TAAG (Dimension-Aware Gravity) algorithm to autonomously segment raw spatial reasoning transcripts into Educational Diagnostic Units (EDUs) and maps them to MJ-DINA feature vectors.",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    # 启动本地服务器，生成一个可访问的 Web 链接
    demo.launch(server_name="0.0.0.0", server_port=7860)