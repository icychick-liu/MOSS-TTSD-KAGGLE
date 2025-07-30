"""
Kaggle Notebook测试代码
直接在Kaggle Notebook中运行的完整测试示例
"""

# ================================
# Cell 1: 环境检查
# ================================
import os
import sys
import torch
import json
import warnings
warnings.filterwarnings("ignore")

print("🔍 Kaggle环境检查")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ================================
# Cell 2: 运行环境设置
# ================================
print("🔧 设置Kaggle环境...")

# 如果kaggle_setup.py存在，运行它
if os.path.exists('kaggle_setup.py'):
    exec(open('kaggle_setup.py').read())
else:
    print("⚠️ 未找到kaggle_setup.py，手动安装依赖...")
    
    # 手动安装关键依赖
    import subprocess
    packages = [
        "transformers==4.53.2",
        "accelerate>=0.20.0", 
        "soundfile",
        "librosa",
        "liger_kernel",
        "pydub"
    ]
    
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            print(f"✅ 安装: {pkg}")
        except:
            print(f"❌ 安装失败: {pkg}")

# ================================
# Cell 3: 创建测试数据
# ================================
print("📝 创建测试数据...")

# 基于examples的测试数据
test_samples = [
    {
        "text": "[S1]Hello! Welcome to MOSS-TTSD-KAGGLE running on Kaggle.[S2]This is incredible! The voice sounds so natural and human-like.[S1]Yes, we're using dual T4 GPUs for acceleration. The quality is amazing!"
    },
    {
        "text": "[S1]你好！欢迎使用在Kaggle上运行的MOSS-TTSD-KAGGLE。[S2]太棒了！声音听起来非常自然，就像真人一样。[S1]是的，我们使用双T4 GPU进行加速，质量非常棒！"
    },
    {
        "text": "[S1]Let me tell you about Context Scaling in AI.[S2]Context Scaling? That sounds fascinating. What exactly does it mean?[S1]It's about helping AI understand complex real-world situations better, not just processing more data."
    }
]

# 保存测试数据
with open("kaggle_test_data.jsonl", "w", encoding="utf-8") as f:
    for sample in test_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ 创建了 {len(test_samples)} 个测试样本")

# ================================
# Cell 4: 运行推理测试
# ================================
print("🎵 开始音频生成测试...")

# 创建输出目录
os.makedirs("kaggle_test_outputs", exist_ok=True)

# 运行推理
import subprocess

cmd = [
    sys.executable, "kaggle_inference.py",
    "--jsonl", "kaggle_test_data.jsonl",
    "--output_dir", "kaggle_test_outputs", 
    "--max_samples", "2",  # 限制样本数量以节省时间
    "--use_normalize",
    "--seed", "42"
]

try:
    print("执行命令:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
    
    print("📊 推理结果:")
    if result.returncode == 0:
        print("✅ 推理成功完成!")
        # 显示输出的最后几行
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines[-10:]:
            if line.strip():
                print(line)
    else:
        print("❌ 推理失败")
        print("错误信息:")
        print(result.stderr)
        
except subprocess.TimeoutExpired:
    print("⏰ 推理超时（30分钟）")
except Exception as e:
    print(f"❌ 运行出错: {e}")

# ================================
# Cell 5: 检查和播放结果
# ================================
print("📊 检查生成结果...")

output_dir = "kaggle_test_outputs"

# 检查生成的文件
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    audio_files = [f for f in files if f.endswith('.wav')]
    
    print(f"🎵 生成的音频文件: {len(audio_files)} 个")
    
    # 显示文件信息
    for audio_file in audio_files:
        file_path = os.path.join(output_dir, audio_file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {audio_file}: {file_size:.1f} KB")
    
    # 读取结果报告
    results_file = os.path.join(output_dir, "kaggle_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        print("\n📋 详细统计:")
        print(f"  总样本数: {results.get('total_samples', 'N/A')}")
        print(f"  成功生成: {results.get('successful_generations', 'N/A')}")
        print(f"  失败数量: {results.get('failed_generations', 'N/A')}")
        
        model_info = results.get('model_info', {})
        print(f"  使用设备: {model_info.get('device', 'N/A')}")
        print(f"  注意力实现: {model_info.get('attention_implementation', 'N/A')}")
        print(f"  文本规范化: {model_info.get('use_normalize', 'N/A')}")
    
    # 在Notebook中播放音频（如果在Jupyter环境中）
    try:
        from IPython.display import Audio, display
        
        print("\n🎧 播放生成的音频:")
        for i, audio_file in enumerate(audio_files[:3]):  # 最多播放3个
            file_path = os.path.join(output_dir, audio_file)
            print(f"播放音频 {i+1}: {audio_file}")
            display(Audio(file_path))
            
    except ImportError:
        print("💡 在Jupyter Notebook中可以直接播放音频")
        print("💡 请下载音频文件到本地播放")

else:
    print("❌ 未找到输出目录")

# ================================
# Cell 6: 性能监控
# ================================
print("📈 性能监控...")

# GPU内存使用情况
if torch.cuda.is_available():
    print("GPU内存使用:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB")

# 系统内存
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"系统内存: {memory.percent}% 使用中")
except ImportError:
    print("无法检查系统内存")

# 清理GPU内存
torch.cuda.empty_cache()
import gc
gc.collect()
print("🧹 GPU内存已清理")

# ================================
# Cell 7: 创建下载包
# ================================
print("📦 创建下载包...")

import zipfile

def create_results_package():
    """创建包含所有结果的压缩包"""
    zip_path = "moss_ttsd_kaggle_test_results.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 添加音频文件
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith(('.wav', '.json')):
                    file_path = os.path.join(output_dir, file)
                    zipf.write(file_path, f"results/{file}")
        
        # 添加测试数据
        if os.path.exists("kaggle_test_data.jsonl"):
            zipf.write("kaggle_test_data.jsonl", "test_data.jsonl")
    
    return zip_path

# 创建下载包
if os.path.exists(output_dir):
    zip_file = create_results_package()
    if os.path.exists(zip_file):
        zip_size = os.path.getsize(zip_file) / 1024  # KB
        print(f"✅ 创建下载包: {zip_file} ({zip_size:.1f} KB)")
    else:
        print("❌ 创建下载包失败")

# ================================
# Cell 8: 测试总结和建议
# ================================
print("\n" + "="*50)
print("🎯 测试总结")
print("="*50)

# 检查测试是否成功
success_indicators = [
    os.path.exists(output_dir),
    len([f for f in os.listdir(output_dir) if f.endswith('.wav')]) > 0 if os.path.exists(output_dir) else False,
    os.path.exists(os.path.join(output_dir, "kaggle_results.json")) if os.path.exists(output_dir) else False
]

if all(success_indicators):
    print("🎉 测试成功完成！")
    print("✅ 环境设置正确")
    print("✅ 模型加载成功") 
    print("✅ 音频生成成功")
    print("✅ 结果保存成功")
else:
    print("⚠️ 测试部分成功或失败")
    print("请检查上面的错误信息")

print("\n💡 使用建议:")
print("1. 根据GPU内存调整--max_samples参数")
print("2. 使用--use_normalize提高文本处理质量")
print("3. 监控Kaggle的9小时使用限制")
print("4. 及时下载生成的音频文件")
print("5. 对于长文本，考虑分段处理")

print("\n🔗 下一步:")
print("- 使用自己的数据替换测试数据")
print("- 调整参数优化生成质量")
print("- 尝试声音克隆功能（需要参考音频）")
print("- 批量处理更多样本")

print("\n📞 如果遇到问题:")
print("- 检查GPU内存是否充足")
print("- 确认所有依赖都已安装")
print("- 查看详细的错误日志")
print("- 尝试减少批处理大小")

print("="*50)
print("🚀 MOSS-TTSD-KAGGLE 测试完成！")
print("="*50)