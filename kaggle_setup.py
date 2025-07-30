#!/usr/bin/env python3
"""
Kaggle环境设置脚本
用于在Kaggle Notebook中部署MOSS-TTSD-KAGGLE项目
"""

import os
import subprocess
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

def install_dependencies():
    """安装必要的依赖包"""
    print("🔧 安装依赖包...")
    
    # 基础依赖
    packages = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "transformers==4.53.2",
        "gradio>=4.0.0",
        "numpy>=1.21.0",
        "accelerate>=0.20.0",
        "soundfile",
        "librosa",
        "tqdm",
        "PyYAML",
        "einops",
        "huggingface_hub",
        "liger_kernel",
        "pydub"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ 已安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {package} - {e}")
    
    # 尝试安装flash-attn (可能失败，但不影响基本功能)
    try:
        print("🔧 尝试安装flash-attn...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "flash-attn", "--no-build-isolation", "--quiet"
        ])
        print("✅ flash-attn安装成功")
    except subprocess.CalledProcessError:
        print("⚠️ flash-attn安装失败，将使用备用attention实现")

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("🏗️ 设置Kaggle环境...")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ 未检测到GPU，将使用CPU模式")
    
    # 创建必要的目录
    directories = ["outputs", "XY_Tokenizer/weights", "temp_audio"]
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

def download_model_weights():
    """下载模型权重文件"""
    print("📥 下载模型权重...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # 下载XY_Tokenizer权重
        weight_path = hf_hub_download(
            repo_id="fnlp/XY_Tokenizer_TTSD_V0",
            filename="xy_tokenizer.ckpt",
            local_dir="./XY_Tokenizer/weights/"
        )
        print(f"✅ XY_Tokenizer权重下载完成: {weight_path}")
        
        return True
    except Exception as e:
        print(f"❌ 模型权重下载失败: {e}")
        return False

def check_system_resources():
    """检查系统资源"""
    print("📊 系统资源检查...")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    except ImportError:
        print("⚠️ 无法检查系统内存")
    
    # 检查磁盘空间
    import shutil
    disk_usage = shutil.disk_usage(".")
    print(f"💿 磁盘空间: {disk_usage.free / 1024**3:.1f}GB 可用")

def main():
    """主函数"""
    print("🚀 开始设置MOSS-TTSD-KAGGLE环境...")
    print("=" * 50)
    
    # 1. 安装依赖
    install_dependencies()
    print()
    
    # 2. 设置环境
    setup_kaggle_environment()
    print()
    
    # 3. 检查系统资源
    check_system_resources()
    print()
    
    # 4. 下载模型权重
    success = download_model_weights()
    print()
    
    if success:
        print("🎉 Kaggle环境设置完成！")
        print("📝 接下来可以运行推理脚本了")
    else:
        print("⚠️ 环境设置完成，但模型权重下载失败")
        print("💡 请手动下载权重文件或检查网络连接")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
