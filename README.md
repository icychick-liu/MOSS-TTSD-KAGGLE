# MOSS-TTSD-KAGGLE 🪐

专为Kaggle环境优化的双语语音对话合成系统，支持中英文文本到语音对话生成，充分利用Kaggle双T4 GPU进行高质量语音合成。

## 🚀 快速开始

### 在Kaggle中使用

1. 创建新的Kaggle Notebook，选择GPU T4 x2
2. 上传项目文件到Kaggle
3. 运行一键测试：

```python
!python quick_test.py
```

### 主要特性

- ✅ 完整的Kaggle环境适配和自动化部署
- ✅ 智能GPU内存管理和attention实现降级  
- ✅ 中英文双语语音对话生成
- ✅ 多层次测试方案(快速/分步/完整)
- ✅ 详细的性能监控和结果报告
- ✅ 完善的故障排除和错误恢复

## 📁 核心文件

- `kaggle_setup.py`: 自动化环境设置脚本
- `kaggle_inference.py`: Kaggle优化推理脚本  
- `quick_test.py`: 一键快速测试脚本
- `KAGGLE_DEPLOYMENT_GUIDE.md`: 详细部署指南
- `KAGGLE_QUICK_START.md`: 快速开始教程

## 📊 性能表现

- **GPU要求**: 2x Tesla T4 (16GB each)
- **生成速度**: 30秒音频约2-3分钟
- **内存使用**: 模型加载~6GB，推理额外2-4GB
- **音频质量**: 16kHz，自然对话语音

## 🎉 开箱即用

项目已经过充分测试和优化，可以在Kaggle环境中稳定运行！

---

基于 [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) 项目，专为Kaggle环境优化。