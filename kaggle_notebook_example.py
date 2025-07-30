"""
Kaggle Notebook示例代码
在Kaggle环境中运行MOSS-TTSD-KAGGLE的完整示例
"""

# ================================
# Cell 1: 环境检查和基础设置
# ================================

import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

print("🔍 检查Kaggle环境...")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")

# 检查GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ 未检测到GPU")

# ================================
# Cell 2: 安装依赖和设置环境
# ================================

# 运行环境设置脚本
print("🔧 设置Kaggle环境...")
exec(open('kaggle_setup.py').read())

# ================================
# Cell 3: 创建测试数据
# ================================

import json

# 创建测试数据
test_data = [
    {
        "text": "[S1]Hello! Welcome to MOSS-TTSD-KAGGLE running on Kaggle.[S2]This is amazing! The voice quality is so natural.[S1]Yes, we're using dual T4 GPUs for acceleration."
    },
    {
        "text": "[S1]你好！欢迎使用在Kaggle上运行的MOSS-TTSD-KAGGLE。[S2]太棒了！语音质量非常自然。[S1]是的，我们使用双T4 GPU进行加速。"
    },
    {
        "text": "[S1]Let me tell you about artificial intelligence.[S2]I'm very interested! What's the latest development?[S1]Well, there's this new concept called Context Scaling..."
    }
]

# 保存测试数据
with open("kaggle_test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"📝 创建了 {len(test_data)} 个测试样本")

# ================================
# Cell 4: 运行推理
# ================================

# 基础推理
print("🎵 开始音频生成...")
os.system("python kaggle_inference.py --jsonl kaggle_test.jsonl --output_dir kaggle_outputs --max_samples 3 --use_normalize")

# ================================
# Cell 5: 查看结果
# ================================

import json
from IPython.display import Audio, display
import os

# 读取结果报告
if os.path.exists("kaggle_outputs/kaggle_results.json"):
    with open("kaggle_outputs/kaggle_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print("📊 生成结果统计:")
    print(f"   总样本数: {results['total_samples']}")
    print(f"   成功生成: {results['successful_generations']}")
    print(f"   失败数量: {results['failed_generations']}")
    print(f"   使用设备: {results['model_info']['device']}")
    print(f"   注意力实现: {results['model_info']['attention_implementation']}")
    
    print("\n🎵 生成的音频文件:")
    for result in results['results']:
        print(f"   文件 {result['index']}: {result['file']} ({result['duration']})")
        
        # 在Notebook中播放音频
        if os.path.exists(result['file']):
            print(f"播放音频 {result['index']}:")
            display(Audio(result['file']))
else:
    print("❌ 未找到结果文件")

# ================================
# Cell 6: 高级用法示例
# ================================

# 带声音克隆的示例（如果有参考音频）
advanced_data = [
    {
        "base_path": "examples",  # 如果上传了examples文件夹
        "text": "[S1]This is a voice cloning test.[S2]The cloned voice sounds very natural!",
        "prompt_audio_speaker1": "m1.wav",  # 需要上传参考音频
        "prompt_text_speaker1": "Reference text for speaker 1",
        "prompt_audio_speaker2": "m2.wav",
        "prompt_text_speaker2": "Reference text for speaker 2"
    }
]

# 如果有参考音频文件，可以运行这个
# with open("advanced_test.jsonl", "w", encoding="utf-8") as f:
#     for item in advanced_data:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")
# 
# os.system("python kaggle_inference.py --jsonl advanced_test.jsonl --output_dir advanced_outputs --max_samples 1")

# ================================
# Cell 7: 性能监控和清理
# ================================

# 检查GPU内存使用
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i} 内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# 清理内存
torch.cuda.empty_cache()
import gc
gc.collect()

print("🧹 内存清理完成")

# ================================
# Cell 8: 下载结果文件
# ================================

# 创建下载包
import zipfile

def create_download_package():
    """创建包含所有生成音频的压缩包"""
    zip_path = "moss_ttsd_kaggle_results.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 添加音频文件
        if os.path.exists("kaggle_outputs"):
            for file in os.listdir("kaggle_outputs"):
                if file.endswith(".wav") or file.endswith(".json"):
                    file_path = os.path.join("kaggle_outputs", file)
                    zipf.write(file_path, file)
        
        # 添加结果报告
        if os.path.exists("kaggle_outputs/kaggle_results.json"):
            zipf.write("kaggle_outputs/kaggle_results.json", "results.json")
    
    print(f"📦 创建下载包: {zip_path}")
    return zip_path

# 创建下载包
download_file = create_download_package()

# 在Kaggle中，你可以通过以下方式下载文件：
print("💾 下载说明:")
print("1. 在Kaggle Notebook中，点击右侧的'Output'标签")
print("2. 找到生成的音频文件和压缩包")
print("3. 点击下载按钮保存到本地")

# ================================
# Cell 9: 使用技巧和注意事项
# ================================

print("💡 使用技巧:")
print("1. Kaggle有9小时的使用限制，请合理安排时间")
print("2. 建议分批处理大量数据，每批不超过5个样本")
print("3. 及时下载生成的音频文件，避免丢失")
print("4. 如果遇到内存不足，减少max_samples参数")
print("5. 使用文本规范化可以提高生成质量")

print("\n⚠️ 注意事项:")
print("1. 确保输入文本使用正确的说话人标签[S1][S2]")
print("2. 音频文件建议控制在60秒以内")
print("3. 参考音频质量会影响声音克隆效果")
print("4. 中英文混合文本可能需要特殊处理")

print("\n🎉 Kaggle部署完成！享受高质量的语音合成吧！")