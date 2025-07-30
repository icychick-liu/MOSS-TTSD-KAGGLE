# MOSS-TTSD-KAGGLE 快速开始指南

## 🎯 一键测试方案

我为你准备了多种测试方案，从简单到复杂，你可以根据需要选择：

### 方案1: 超级快速测试 ⚡ (推荐新手)

在Kaggle Notebook中运行一个cell：

```python
# 下载并运行快速测试
!python quick_test.py
```

这个脚本会自动：
- ✅ 安装所有依赖
- ✅ 下载模型权重  
- ✅ 创建测试数据
- ✅ 运行推理生成音频
- ✅ 检查结果并创建下载包

### 方案2: 分步测试 📋 (推荐进阶用户)

```python
# 步骤1: 环境设置
!python kaggle_setup.py

# 步骤2: 运行测试
!python kaggle_test_runner.py

# 步骤3: 检查结果
import os
print("生成的文件:", os.listdir("test_outputs_text_only"))
```

### 方案3: 完整Notebook体验 📓 (推荐完整体验)

将 `kaggle_notebook_test.py` 的内容复制到Kaggle Notebook的多个cell中运行。

## 🚀 在Kaggle中的具体操作步骤

### 第一步：创建Kaggle Notebook

1. 登录 [Kaggle](https://www.kaggle.com)
2. 点击 "Create" → "New Notebook"
3. **重要**: 选择 "GPU T4 x2" 作为加速器
4. 设置为 "Internet On"

### 第二步：上传项目文件

创建一个新的Dataset或直接在Notebook中上传以下文件：

```
必需文件:
├── kaggle_setup.py              # 环境设置脚本
├── kaggle_inference.py          # 推理脚本  
├── quick_test.py               # 快速测试脚本
├── generation_utils.py         # 生成工具
├── modeling_asteroid.py        # 模型定义
└── XY_Tokenizer/              # 分词器模块
    ├── config/
    │   └── xy_tokenizer_config.yaml
    └── xy_tokenizer/
        ├── model.py
        └── nn/

可选文件:
├── examples/                   # 示例音频文件
│   ├── zh_spk1_moon.wav
│   ├── zh_spk2_moon.wav  
│   ├── m1.wav
│   └── m2.wav
└── kaggle_test_runner.py      # 完整测试脚本
```

### 第三步：运行测试

在Notebook的第一个cell中：

```python
# 方案1: 一键测试（最简单）
!python quick_test.py
```

或者

```python
# 方案2: 分步测试（更可控）
# Cell 1: 环境设置
!python kaggle_setup.py

# Cell 2: 创建测试数据并运行
import json

# 创建简单测试数据
test_data = [
    {"text": "[S1]Hello! Testing MOSS-TTSD-KAGGLE on Kaggle.[S2]This sounds amazing![S1]Yes, very natural!"},
    {"text": "[S1]你好！在Kaggle上测试MOSS-TTSD-KAGGLE。[S2]听起来太棒了！[S1]是的，非常自然！"}
]

with open("my_test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Cell 3: 运行推理
!python kaggle_inference.py --jsonl my_test.jsonl --output_dir my_outputs --max_samples 2 --use_normalize

# Cell 4: 检查结果
import os
from IPython.display import Audio, display

if os.path.exists("my_outputs"):
    audio_files = [f for f in os.listdir("my_outputs") if f.endswith('.wav')]
    print(f"生成了 {len(audio_files)} 个音频文件")
    
    # 播放音频
    for audio_file in audio_files:
        print(f"播放: {audio_file}")
        display(Audio(f"my_outputs/{audio_file}"))
```

## 📊 预期结果

### 成功指标
- ✅ 环境设置无错误
- ✅ 模型权重下载成功
- ✅ 生成至少1个音频文件
- ✅ 音频文件大小 > 0KB
- ✅ 生成结果报告文件

### 性能预期
- **设置时间**: 5-10分钟
- **单个样本生成时间**: 2-5分钟
- **GPU内存使用**: 6-8GB
- **生成音频质量**: 16kHz, 自然对话语音

## 🔧 常见问题解决

### 问题1: Flash Attention安装失败
```
⚠️ flash-attn安装失败，将使用备用attention实现
```
**解决**: 这是正常的，脚本会自动使用其他attention实现。

### 问题2: 模型下载慢或失败
```python
# 手动下载模型权重
from huggingface_hub import hf_hub_download
import os

os.makedirs("XY_Tokenizer/weights", exist_ok=True)
hf_hub_download(
    repo_id="fnlp/XY_Tokenizer_TTSD_V0",
    filename="xy_tokenizer.ckpt", 
    local_dir="./XY_Tokenizer/weights/"
)
```

### 问题3: GPU内存不足
```python
# 减少批处理大小
!python kaggle_inference.py --max_samples 1

# 清理GPU内存
import torch
torch.cuda.empty_cache()
```

### 问题4: 生成的音频文件为空
- 检查输入文本格式是否正确（使用[S1][S2]标签）
- 确认GPU可用且模型加载成功
- 查看详细错误日志

## 💡 优化建议

### 提高生成质量
```python
# 使用文本规范化
!python kaggle_inference.py --use_normalize

# 设置随机种子保证可重复性
!python kaggle_inference.py --seed 42
```

### 节省资源
```python
# 限制样本数量
!python kaggle_inference.py --max_samples 3

# 使用较短的文本（建议 < 100字）
```

### 批量处理
```python
# 分批处理大量数据
for i in range(0, total_samples, batch_size):
    batch_file = f"batch_{i}.jsonl"
    # 创建批次文件...
    !python kaggle_inference.py --jsonl {batch_file} --output_dir batch_{i}_outputs
```

## 📁 输出文件说明

生成的文件结构：
```
outputs/
├── kaggle_output_0.wav        # 生成的音频文件
├── kaggle_output_1.wav
├── ...
└── kaggle_results.json        # 详细结果报告

moss_ttsd_quick_test_results.zip  # 打包下载文件
```

结果报告包含：
- 处理统计信息
- 每个文件的详细信息  
- 模型配置参数
- 性能指标

## ⏰ 时间管理

Kaggle有9小时使用限制，建议：
- 小批量测试：2-3个样本，用时10-15分钟
- 中等批量：5-10个样本，用时30-60分钟  
- 大批量：分多次session处理

## 🎵 音频质量检查

生成的音频应该具有：
- 清晰的语音质量
- 自然的对话节奏
- 正确的说话人切换
- 16kHz采样率
- 单声道格式

## 📞 获取帮助

如果遇到问题：
1. 检查Kaggle Notebook的完整输出日志
2. 确认GPU加速已启用（右侧面板显示GPU使用情况）
3. 验证所有必需文件都已上传
4. 尝试重启Notebook并重新运行

---

**开始你的MOSS-TTSD-KAGGLE之旅吧！🚀**
