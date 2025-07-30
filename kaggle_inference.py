#!/usr/bin/env python3
"""
Kaggle优化版推理脚本
适配Kaggle环境的MOSS-TTSD-KAGGLE推理
"""

import json
import torch
import torchaudio
import accelerate
import argparse
import os
import warnings
import gc
from pathlib import Path

# 忽略警告
warnings.filterwarnings("ignore")

# 尝试导入项目模块
try:
    from generation_utils import load_model, process_batch
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 请确保所有项目文件都在当前目录中")
    exit(1)

# Kaggle环境配置
KAGGLE_CONFIG = {
    "MODEL_PATH": "fnlp/MOSS-TTSD-v0.5",
    "SYSTEM_PROMPT": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    "SPT_CONFIG_PATH": "XY_Tokenizer/config/xy_tokenizer_config.yaml",
    "SPT_CHECKPOINT_PATH": "XY_Tokenizer/weights/xy_tokenizer.ckpt",
    "MAX_CHANNELS": 8,
    "DEFAULT_OUTPUT_DIR": "outputs",
    "TEMP_DIR": "temp_audio"
}

def check_kaggle_environment():
    """检查Kaggle环境"""
    print("🔍 检查Kaggle环境...")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
    else:
        print("⚠️ 未检测到GPU，将使用CPU模式（速度较慢）")
        return False

def optimize_for_kaggle():
    """Kaggle环境优化"""
    print("⚡ 优化Kaggle环境...")
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # 优化PyTorch设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建必要目录
    for dir_name in [KAGGLE_CONFIG["DEFAULT_OUTPUT_DIR"], KAGGLE_CONFIG["TEMP_DIR"]]:
        os.makedirs(dir_name, exist_ok=True)

def load_models_with_fallback():
    """加载模型（带降级处理）"""
    print("📦 加载模型...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 尝试不同的attention实现
    attention_implementations = ["flash_attention_2", "sdpa", "eager"]
    
    for attn_impl in attention_implementations:
        try:
            print(f"🔧 尝试使用 {attn_impl} attention...")
            
            tokenizer, model, spt = load_model(
                KAGGLE_CONFIG["MODEL_PATH"], 
                KAGGLE_CONFIG["SPT_CONFIG_PATH"], 
                KAGGLE_CONFIG["SPT_CHECKPOINT_PATH"],
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl
            )
            
            spt = spt.to(device)
            model = model.to(device)
            
            print(f"✅ 模型加载成功 (使用 {attn_impl})")
            return tokenizer, model, spt, device, attn_impl
            
        except Exception as e:
            print(f"❌ {attn_impl} 失败: {e}")
            continue
    
    raise RuntimeError("所有attention实现都失败了")

def create_sample_data():
    """创建示例数据（如果没有输入文件）"""
    sample_data = [
        {
            "text": "[S1]Hello, this is a test of the MOSS-TTSD-KAGGLE system.[S2]Yes, it sounds very natural and realistic![S1]I'm glad you think so. This is running on Kaggle with dual T4 GPUs."
        },
        {
            "text": "[S1]你好，这是MOSS-TTSD-KAGGLE系统的测试。[S2]是的，听起来非常自然和真实！[S1]很高兴你这么认为。这是在Kaggle上使用双T4 GPU运行的。"
        }
    ]
    
    # 保存示例数据
    sample_file = "kaggle_sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"📝 创建示例数据文件: {sample_file}")
    return sample_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Kaggle优化版TTS推理")
    parser.add_argument("--jsonl", default=None, help="输入JSONL文件路径")
    parser.add_argument("--output_dir", default=KAGGLE_CONFIG["DEFAULT_OUTPUT_DIR"], help="输出目录")
    parser.add_argument("--use_normalize", action="store_true", default=True, help="使用文本规范化")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_samples", type=int, default=5, help="最大处理样本数（Kaggle限制）")
    
    args = parser.parse_args()
    
    print("🚀 启动Kaggle版MOSS-TTSD-KAGGLE推理...")
    print("=" * 60)
    
    # 1. 检查环境
    has_gpu = check_kaggle_environment()
    
    # 2. 优化环境
    optimize_for_kaggle()
    
    # 3. 处理输入文件
    if args.jsonl is None or not os.path.exists(args.jsonl):
        print("⚠️ 未指定输入文件或文件不存在，使用示例数据")
        args.jsonl = create_sample_data()
    
    # 4. 加载数据
    try:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            items = [json.loads(line.strip()) for line in f if line.strip()]
        
        # 限制样本数量（Kaggle资源限制）
        if len(items) > args.max_samples:
            print(f"⚠️ 样本数量过多，限制为前{args.max_samples}个")
            items = items[:args.max_samples]
        
        print(f"📊 加载了 {len(items)} 个样本")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 5. 加载模型
    try:
        tokenizer, model, spt, device, attn_impl = load_models_with_fallback()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 6. 设置随机种子
    if args.seed is not None:
        accelerate.utils.set_seed(args.seed)
        print(f"🎲 设置随机种子: {args.seed}")
    
    # 7. 开始推理
    print("🎵 开始音频生成...")
    try:
        actual_texts_data, audio_results = process_batch(
            batch_items=items,
            tokenizer=tokenizer,
            model=model,
            spt=spt,
            device=device,
            system_prompt=KAGGLE_CONFIG["SYSTEM_PROMPT"],
            start_idx=0,
            use_normalize=args.use_normalize
        )
        
        # 8. 保存结果
        saved_count = 0
        results_info = []
        
        for idx, audio_result in enumerate(audio_results):
            if audio_result is not None:
                output_path = os.path.join(args.output_dir, f"kaggle_output_{idx}.wav")
                
                try:
                    torchaudio.save(
                        output_path,
                        audio_result["audio_data"],
                        audio_result["sample_rate"]
                    )
                    
                    # 计算音频时长
                    duration = audio_result["audio_data"].shape[-1] / audio_result["sample_rate"]
                    
                    results_info.append({
                        "index": idx,
                        "file": output_path,
                        "duration": f"{duration:.2f}s",
                        "sample_rate": audio_result["sample_rate"]
                    })
                    
                    print(f"✅ 保存音频 {idx}: {output_path} ({duration:.2f}s)")
                    saved_count += 1
                    
                except Exception as e:
                    print(f"❌ 保存音频 {idx} 失败: {e}")
            else:
                print(f"⚠️ 跳过样本 {idx}（生成失败）")
        
        # 9. 生成结果报告
        report_path = os.path.join(args.output_dir, "kaggle_results.json")
        report = {
            "total_samples": len(items),
            "successful_generations": saved_count,
            "failed_generations": len(items) - saved_count,
            "model_info": {
                "model_path": KAGGLE_CONFIG["MODEL_PATH"],
                "attention_implementation": attn_impl,
                "device": device,
                "use_normalize": args.use_normalize
            },
            "results": results_info
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 结果报告保存至: {report_path}")
        
        # 10. 清理内存
        del model, spt, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        print("=" * 60)
        print(f"🎉 推理完成！成功生成 {saved_count}/{len(items)} 个音频文件")
        print(f"📁 输出目录: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 推理过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
