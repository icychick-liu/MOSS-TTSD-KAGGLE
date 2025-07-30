# MOSS-TTSD-KAGGLE 部署指南

## 🎯 概述

本指南将帮助你在Kaggle环境中部署MOSS-TTSD-KAGGLE文本到语音对话生成系统，充分利用Kaggle提供的2个T4 GPU进行加速推理。

## 📋 前置要求

- Kaggle账户（已验证手机号）
- 启用GPU加速的Kaggle Notebook
- 基本的Python和机器学习知识

## 🚀 快速开始

### 步骤1：创建Kaggle Notebook

1. 登录Kaggle，点击"Create" → "New Notebook"
2. 选择"GPU T4 x2"作为加速器
3. 设置Notebook为"Public"或"Private"

### 步骤2：上传项目文件

将以下文件上传到Kaggle Dataset或直接复制到Notebook中：

```
MOSS-TTSD/
├── generation_utils.py          # 核心生成工具
├── modeling_asteroid.py         # 模型定义
├── kaggle_setup.py             # Kaggle环境设置脚本
├── kaggle_inference.py         # Kaggle优化推理脚本
├── XY_Tokenizer/               # 分词器模块
│   ├── config/
│   │   └── xy_tokenizer_config.yaml
│   └── xy_tokenizer/
│       ├── model.py
│       └── nn/
└── examples/                   # 示例文件（可选）
```

### 步骤3：环境设置

在Kaggle Notebook的第一个cell中运行：

```python
# 安装依赖和设置环境
!python kaggle_setup.py
```

这个脚本会：
- ✅ 安装所有必要的Python包
- ✅ 下载模型权重文件
- ✅ 检查GPU可用性
- ✅ 创建必要的目录结构

### 步骤4：运行推理

```python
# 基础推理（使用内置示例）
!python kaggle_inference.py

# 使用自定义数据
!python kaggle_inference.py --jsonl your_data.jsonl --output_dir outputs --max_samples 3
```

## 📊 性能优化建议

### GPU内存优化

```python
# 在推理前清理GPU内存
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
```

### 批处理大小调整

由于Kaggle的资源限制，建议：
- 单次处理样本数：≤ 5个
- 音频长度：≤ 60秒
- 使用`--max_samples`参数控制处理数量

### 注意力机制降级

脚本会自动尝试以下attention实现：
1. `flash_attention_2`（最优）
2. `sdpa`（备选）
3. `eager`（兜底）

## 📝 输入数据格式

### 格式1：纯文本（无声音克隆）
```json
{
  "text": "[S1]Speaker 1 content[S2]Speaker 2 content[S1]..."
}
```

### 格式2：带声音克隆
```json
{
  "base_path": "audio_files/",
  "text": "[S1]对话内容[S2]回复内容",
  "prompt_audio_speaker1": "speaker1.wav",
  "prompt_text_speaker1": "说话人1的参考文本",
  "prompt_audio_speaker2": "speaker2.wav",
  "prompt_text_speaker2": "说话人2的参考文本"
}
```

## 🔧 故障排除

### 常见问题

**1. Flash Attention安装失败**
```
⚠️ flash-attn安装失败，将使用备用attention实现
```
- 这是正常的，脚本会自动降级到其他实现

**2. 模型权重下载失败**
```python
# 手动下载权重
from huggingface_hub import hf_hub_download

weight_path = hf_hub_download(
    repo_id="fnlp/XY_Tokenizer_TTSD_V0",
    filename="xy_tokenizer.ckpt",
    local_dir="./XY_Tokenizer/weights/"
)
```

**3. GPU内存不足**
```python
# 减少批处理大小
!python kaggle_inference.py --max_samples 1
```

**4. 音频文件过大**
- 限制音频长度 ≤ 30秒
- 使用较低的采样率

### 性能监控

```python
# 检查GPU使用情况
!nvidia-smi

# 检查内存使用
import psutil
print(f"内存使用: {psutil.virtual_memory().percent}%")
```

## 📈 预期性能

### 硬件配置
- **GPU**: 2x Tesla T4 (16GB each)
- **RAM**: ~13GB
- **存储**: ~20GB

### 生成速度
- **短对话** (30秒): ~2-3分钟
- **中等对话** (60秒): ~4-6分钟
- **长对话** (120秒): ~8-12分钟

### 内存使用
- **模型加载**: ~6GB GPU内存
- **推理过程**: ~2-4GB额外内存
- **音频生成**: 根据长度线性增长

## 🎵 输出文件

生成的文件将保存在`outputs/`目录：

```
outputs/
├── kaggle_output_0.wav        # 生成的音频文件
├── kaggle_output_1.wav
├── ...
└── kaggle_results.json        # 结果报告
```

结果报告包含：
- 处理统计信息
- 每个文件的详细信息
- 模型配置信息
- 错误日志

## 💡 最佳实践

### 1. 数据准备
- 使用UTF-8编码的JSONL文件
- 确保音频文件格式为WAV
- 预处理文本，移除特殊字符

### 2. 资源管理
- 定期清理GPU内存
- 监控磁盘空间使用
- 及时下载生成的音频文件

### 3. 质量优化
- 启用文本规范化（`--use_normalize`）
- 提供高质量的参考音频
- 使用合适的说话人标签

## 🔗 相关链接

- [MOSS-TTSD GitHub](https://github.com/OpenMOSS/MOSS-TTSD)
- [Hugging Face模型页面](https://huggingface.co/fnlp/MOSS-TTSD-v0.5)
- [Kaggle GPU文档](https://www.kaggle.com/docs/efficient-gpu-usage)

## 📞 技术支持

如果遇到问题：
1. 检查Kaggle Notebook的输出日志
2. 确认GPU加速已启用
3. 验证所有文件都已正确上传
4. 参考故障排除部分

---

**注意**: Kaggle有使用时间限制（通常为9小时），请合理安排推理任务。建议分批处理大量数据，并及时保存结果。
