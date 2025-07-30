#!/usr/bin/env python3
"""
Kaggle测试运行器
使用examples中的示例数据测试kaggle_inference.py
"""

import json
import os
import sys
import subprocess
from pathlib import Path

def create_test_data():
    """创建基于examples的测试数据"""
    print("📝 创建测试数据...")
    
    # 测试数据集1：纯文本模式（无声音克隆）
    text_only_data = [
        {
            "text": "[S1]Hello! This is a test of MOSS-TTSD-KAGGLE running on Kaggle with dual T4 GPUs.[S2]Wow, that sounds amazing! The voice quality is incredibly natural.[S1]Yes, we're using the latest text-to-speech technology for realistic dialogue generation."
        },
        {
            "text": "[S1]你好！这是在Kaggle上使用双T4 GPU运行的MOSS-TTSD-KAGGLE测试。[S2]哇，听起来太棒了！语音质量非常自然。[S1]是的，我们使用最新的文本转语音技术来生成真实的对话。"
        },
        {
            "text": "[S1]Let me tell you about Context Scaling in AI.[S2]Context Scaling? That sounds interesting. What does it mean?[S1]It's a new concept that focuses on helping AI understand complex, real-world situations better."
        }
    ]
    
    # 测试数据集2：带声音克隆（如果有参考音频）
    voice_cloning_data = [
        {
            "base_path": "examples",
            "text": "[S1]诶，我最近看了一篇讲人工智能的文章，还挺有意思的，想跟你聊聊。[S2]哦？是吗，关于啥的啊？又是哪个公司发了什么逆天的新模型吗？[S1]那倒不是，是一个咱们国内的教授，复旦大学的邱锡鹏教授，他提了一个新概念，叫情境扩展。",
            "prompt_audio_speaker1": "zh_spk1_moon.wav",
            "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段。",
            "prompt_audio_speaker2": "zh_spk2_moon.wav", 
            "prompt_text_speaker2": "如果大家想听到更丰富更及时的直播内容，记得准时进入直播间。"
        },
        {
            "base_path": "examples",
            "text": "[S1]Hey, did you hear about that company called MoSi AI?[S2]MoSi AI? Yeah, I think I've heard of them. What new thing have they come up with now?[S1]They recently launched this super hot new product called Asteroid.",
            "prompt_audio_speaker1": "m1.wav",
            "prompt_text_speaker1": "How much do you know about her?",
            "prompt_audio_speaker2": "m2.wav",
            "prompt_text_speaker2": "Well, we know this much about her. You've been with her constantly."
        }
    ]
    
    # 保存测试数据
    test_files = {}
    
    # 纯文本测试
    with open("test_text_only.jsonl", "w", encoding="utf-8") as f:
        for item in text_only_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    test_files["text_only"] = "test_text_only.jsonl"
    print(f"✅ 创建纯文本测试文件: {test_files['text_only']} ({len(text_only_data)} 样本)")
    
    # 声音克隆测试（仅在有音频文件时）
    if os.path.exists("examples"):
        with open("test_voice_cloning.jsonl", "w", encoding="utf-8") as f:
            for item in voice_cloning_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        test_files["voice_cloning"] = "test_voice_cloning.jsonl"
        print(f"✅ 创建声音克隆测试文件: {test_files['voice_cloning']} ({len(voice_cloning_data)} 样本)")
    else:
        print("⚠️ 未找到examples目录，跳过声音克隆测试")
    
    return test_files

def run_inference_test(test_file, output_dir, test_name, max_samples=2):
    """运行推理测试"""
    print(f"\n🎵 开始 {test_name} 测试...")
    print(f"   输入文件: {test_file}")
    print(f"   输出目录: {output_dir}")
    print(f"   最大样本数: {max_samples}")
    
    # 构建命令
    cmd = [
        sys.executable, "kaggle_inference.py",
        "--jsonl", test_file,
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--use_normalize",
        "--seed", "42"
    ]
    
    try:
        # 运行推理
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {test_name} 测试成功完成")
            print("📊 输出摘要:")
            # 显示最后几行输出
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ {test_name} 测试失败")
            print("错误输出:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行 {test_name} 测试时出错: {e}")
        return False

def check_results(output_dir):
    """检查生成结果"""
    print(f"\n📊 检查结果目录: {output_dir}")
    
    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在")
        return False
    
    # 检查音频文件
    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    print(f"🎵 生成的音频文件数量: {len(audio_files)}")
    
    for audio_file in audio_files:
        file_path = os.path.join(output_dir, audio_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   {audio_file}: {file_size:.1f} KB")
    
    # 检查结果报告
    results_file = os.path.join(output_dir, "kaggle_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            print("📋 结果统计:")
            print(f"   总样本数: {results.get('total_samples', 'N/A')}")
            print(f"   成功生成: {results.get('successful_generations', 'N/A')}")
            print(f"   失败数量: {results.get('failed_generations', 'N/A')}")
            print(f"   使用设备: {results.get('model_info', {}).get('device', 'N/A')}")
            print(f"   注意力实现: {results.get('model_info', {}).get('attention_implementation', 'N/A')}")
            
            return True
        except Exception as e:
            print(f"❌ 读取结果文件失败: {e}")
            return False
    else:
        print("⚠️ 未找到结果报告文件")
        return False

def main():
    """主测试函数"""
    print("🚀 开始MOSS-TTSD-KAGGLE测试")
    print("=" * 60)
    
    # 检查必要文件
    required_files = ["kaggle_inference.py", "generation_utils.py", "modeling_asteroid.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保所有项目文件都在当前目录中")
        return
    
    # 创建测试数据
    test_files = create_test_data()
    
    if not test_files:
        print("❌ 无法创建测试数据")
        return
    
    # 运行测试
    test_results = {}
    
    # 测试1: 纯文本模式
    if "text_only" in test_files:
        output_dir = "test_outputs_text_only"
        os.makedirs(output_dir, exist_ok=True)
        
        success = run_inference_test(
            test_files["text_only"], 
            output_dir, 
            "纯文本模式",
            max_samples=2
        )
        test_results["text_only"] = success
        
        if success:
            check_results(output_dir)
    
    # 测试2: 声音克隆模式（如果有音频文件）
    if "voice_cloning" in test_files:
        output_dir = "test_outputs_voice_cloning"
        os.makedirs(output_dir, exist_ok=True)
        
        success = run_inference_test(
            test_files["voice_cloning"], 
            output_dir, 
            "声音克隆模式",
            max_samples=1  # 声音克隆测试用较少样本
        )
        test_results["voice_cloning"] = success
        
        if success:
            check_results(output_dir)
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("🎯 测试结果总结:")
    
    for test_name, success in test_results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    if successful_tests == total_tests:
        print(f"\n🎉 所有测试通过！({successful_tests}/{total_tests})")
        print("💡 你现在可以使用kaggle_inference.py处理自己的数据了")
    else:
        print(f"\n⚠️ 部分测试失败 ({successful_tests}/{total_tests})")
        print("💡 请检查错误信息并确保环境设置正确")
    
    # 提供使用建议
    print("\n📝 使用建议:")
    print("1. 查看生成的音频文件质量")
    print("2. 根据需要调整--max_samples参数")
    print("3. 使用--use_normalize提高文本处理质量")
    print("4. 监控GPU内存使用情况")

if __name__ == "__main__":
    main()