#!/usr/bin/env python3
"""
MOSS-TTSD-KAGGLE 快速测试脚本
一键运行完整测试流程
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """打印步骤"""
    print(f"\n📋 步骤 {step}: {description}")
    print("-" * 40)

def run_command(cmd, description, timeout=1800):
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    print(f"执行: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ 成功完成 ({end_time - start_time:.1f}秒)")
            return True, result.stdout
        else:
            print(f"❌ 失败 (返回码: {result.returncode})")
            print("错误信息:")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时 ({timeout}秒)")
        return False, "命令执行超时"
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False, str(e)

def main():
    """主测试流程"""
    print_header("MOSS-TTSD-KAGGLE 快速测试")
    
    # 检查必要文件
    required_files = [
        "kaggle_setup.py",
        "kaggle_inference.py", 
        "generation_utils.py",
        "modeling_asteroid.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保所有项目文件都在当前目录中")
        return False
    
    print("✅ 所有必要文件都存在")
    
    # 步骤1: 环境设置
    print_step(1, "环境设置和依赖安装")
    success, output = run_command(
        [sys.executable, "kaggle_setup.py"],
        "设置Kaggle环境",
        timeout=600  # 10分钟
    )
    
    if not success:
        print("❌ 环境设置失败，无法继续测试")
        return False
    
    # 步骤2: 创建测试数据
    print_step(2, "创建测试数据")
    
    test_data = [
        {
            "text": "[S1]Hello! This is a quick test of MOSS-TTSD-KAGGLE on Kaggle.[S2]Wow, this is working great with dual T4 GPUs![S1]Yes, the voice quality is amazing and very natural."
        },
        {
            "text": "[S1]你好！这是MOSS-TTSD-KAGGLE在Kaggle上的快速测试。[S2]哇，使用双T4 GPU效果真的很棒！[S1]是的，语音质量非常棒，听起来很自然。"
        }
    ]
    
    test_file = "quick_test_data.jsonl"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ 创建测试数据: {test_file} ({len(test_data)} 样本)")
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        return False
    
    # 步骤3: 运行推理
    print_step(3, "运行音频生成推理")
    
    output_dir = "quick_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    inference_cmd = [
        sys.executable, "kaggle_inference.py",
        "--jsonl", test_file,
        "--output_dir", output_dir,
        "--max_samples", "2",
        "--use_normalize",
        "--seed", "42"
    ]
    
    success, output = run_command(
        inference_cmd,
        "生成音频",
        timeout=1800  # 30分钟
    )
    
    if not success:
        print("❌ 音频生成失败")
        return False
    
    # 步骤4: 检查结果
    print_step(4, "检查生成结果")
    
    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在")
        return False
    
    # 检查音频文件
    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    print(f"🎵 生成的音频文件: {len(audio_files)} 个")
    
    total_size = 0
    for audio_file in audio_files:
        file_path = os.path.join(output_dir, audio_file)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        print(f"   {audio_file}: {file_size / 1024:.1f} KB")
    
    print(f"📊 总文件大小: {total_size / 1024:.1f} KB")
    
    # 检查结果报告
    results_file = os.path.join(output_dir, "kaggle_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            print("📋 生成统计:")
            print(f"   总样本数: {results.get('total_samples', 'N/A')}")
            print(f"   成功生成: {results.get('successful_generations', 'N/A')}")
            print(f"   失败数量: {results.get('failed_generations', 'N/A')}")
            
            model_info = results.get('model_info', {})
            print(f"   使用设备: {model_info.get('device', 'N/A')}")
            print(f"   注意力实现: {model_info.get('attention_implementation', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ 读取结果报告失败: {e}")
    
    # 步骤5: 创建下载包
    print_step(5, "创建结果下载包")
    
    try:
        import zipfile
        
        zip_path = "moss_ttsd_quick_test_results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # 添加音频文件
            for file in os.listdir(output_dir):
                if file.endswith(('.wav', '.json')):
                    file_path = os.path.join(output_dir, file)
                    zipf.write(file_path, f"results/{file}")
            
            # 添加测试数据
            zipf.write(test_file, "test_data.jsonl")
        
        zip_size = os.path.getsize(zip_path) / 1024
        print(f"✅ 创建下载包: {zip_path} ({zip_size:.1f} KB)")
        
    except Exception as e:
        print(f"⚠️ 创建下载包失败: {e}")
    
    # 测试总结
    print_header("测试完成")
    
    success_count = len(audio_files)
    total_samples = len(test_data)
    
    if success_count > 0:
        print("🎉 快速测试成功完成！")
        print(f"✅ 成功生成 {success_count}/{total_samples} 个音频文件")
        print(f"📁 输出目录: {output_dir}")
        print(f"📦 下载包: {zip_path if 'zip_path' in locals() else '未创建'}")
        
        print("\n💡 接下来你可以:")
        print("1. 播放生成的音频文件检查质量")
        print("2. 使用自己的数据替换测试数据")
        print("3. 调整参数优化生成效果")
        print("4. 尝试更多样本或更长的文本")
        
        return True
    else:
        print("❌ 快速测试失败")
        print("请检查上面的错误信息并重试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
