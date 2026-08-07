import os
import re
import pandas as pd
import win32com.client as win32

def extract_interviewee_responses(folder_path, output_csv):
    """
    遍历文件夹中的所有 .doc 文件，提取 Interviewee 的发言并保存为 CSV。
    """
    print(f"📂 正在连接 Microsoft Word 引擎，准备读取目录: {folder_path} ...")
    
    # 启动后台 Word 应用程序 (不可见模式，避免弹窗打扰)
    word = win32.Dispatch("Word.Application")
    word.Visible = False
    
    data = []
    
    # 获取目标文件夹下所有的 .doc / .docx 文件
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.doc') or f.endswith('.docx')]
    print(f"📄 找到 {len(valid_files)} 个文档，开始执行心智切片提取...\n")

    for filename in valid_files:
        # 获取绝对路径 (win32com 必须使用绝对路径)
        file_path = os.path.abspath(os.path.join(folder_path, filename))
        # 提取学生 ID (例如: ET22_0005)
        student_id = os.path.splitext(filename)[0]
        
        try:
            # 1. 打开文档并提取所有纯文本
            doc = word.Documents.Open(file_path)
            full_text = doc.Content.Text
            doc.Close(False) # 关闭文档，不保存更改
            
            # 2. 物理切割：使用正则表达式提取 Interviewee 的所有发言
            # 逻辑：匹配 "Interviewee:" 之后的所有内容，直到遇到下一个 "Interviewer:" 或文件结尾。
            # re.DOTALL 允许跨越多行匹配，re.IGNORECASE 忽略大小写
            pattern = re.compile(r'Interviewee:\s*(.*?)(?=(?:Interviewer:|$))', re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(full_text)
            
            # 3. 数据清洗：去除提取文本中多余的换行、制表符和空格
            cleaned_responses = []
            for match in matches:
                # 将内部的换行符和制表符替换为空格，保持单句连贯
                clean_text = " ".join(match.strip().split())
                if len(clean_text) > 0:
                    cleaned_responses.append(clean_text)
            
            # 4. 汇总策略：你可以选择将所有回合合并，或者用特定符号隔开
            # 这里我们用 " | " 隔开每一次发言，方便后续 SBERT 滑动窗口处理
            combined_response = " | ".join(cleaned_responses)
            
            data.append({
                "ID": student_id,
                "Total_Turns": len(cleaned_responses), # 记录该学生总共发言了几次
                "Interviewee_Response": combined_response
            })
            
            print(f"✅ 成功处理: {student_id} (提取到 {len(cleaned_responses)} 次发言)")
            
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时发生错误: {e}")

    # 退出 Word 引擎释放内存
    word.Quit()
    
    # 将提取的结构化数据转化为 DataFrame 并导出为 CSV
    if data:
        df = pd.DataFrame(data)
        # 使用 utf-8-sig 确保生成的 CSV 在 Excel 中打开时不会有乱码
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n🎉 提取完成！共处理 {len(data)} 份文档，结果已保存至: {os.path.abspath(output_csv)}")
    else:
        print("\n⚠️ 未提取到任何数据，请检查文件夹路径或文档内容格式。")

# ==========================================
# 启动执行区
# ==========================================
if __name__ == "__main__":
    # 替换为你存放 doc 文件的实际文件夹路径 (相对路径或绝对路径均可)
    INPUT_FOLDER = "Think-aloud-All" 
    # 你期望输出的 CSV 文件名
    OUTPUT_CSV = "Extracted_ThinkAloud_Responses.csv"
    
    # 如果文件夹不存在，给出提示
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 找不到文件夹 '{INPUT_FOLDER}'。请确保它与此脚本在同一目录下。")
    else:
        extract_interviewee_responses(INPUT_FOLDER, OUTPUT_CSV)