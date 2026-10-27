# deai：本地英文风格适配与改写管道

**核心功能**

1. **目标风格数据增强**：从自己有权使用的英文文本构造合成源文 → 原文训练对，先分训练/验证集，再做依存树扰动、缩写展开及有限语域替换；质量不达标则拒绝训练。
2. **TinyLlama 风格训练**：在 Apple Silicon 上使用 MLX 训练全部 22 个 decoder block（约 **88%** 参数，非严格 100% 全参）。每次训练保存到独立目录，不覆盖其他用户权重。
3. **三阶段文本处理**：依存扰动 → 微调后的 TinyLlama-1.1B-Chat 生成 → 可选的拼写/常见语法错误控制。第三步默认关闭。

支持两种本地 CLI 流程：**下载示例权重直接试用**，或**提供自己的目标风格文本重新训练后使用**。不需要 API 密钥、在线服务或上传用户文本。它不能保证规避 AI 检测器，也不能保证事实和文风完全一致；请人工核对输出。

## 准备环境

需要 Apple Silicon macOS、Python 3.11，以及足够内存和磁盘空间。本仓库不包含基础模型；从 [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) 下载 `config.json`、权重及 tokenizer 文件到 `model/`。模型许可证请以该模型卡为准。

```bash
git clone https://github.com/kevin0818-lxd/deai.git
cd deai
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir model
```

`model/` 和语言解析器只在安装时下载；文本处理与训练在本机完成。风格和待处理输入均为 UTF-8 英文纯文本。

## 流程 A：使用示例权重

从 [v0.1.0 Release](https://github.com/kevin0818-lxd/deai/releases/tag/v0.1.0) 下载 `adapters.safetensors` 至仓库自带的 `template/`，与 `template/adapter_config.json` 放在同一目录：

```bash
curl -fL https://github.com/kevin0818-lxd/deai/releases/download/v0.1.0/adapters.safetensors \
  -o template/adapters.safetensors
shasum -a 256 template/adapters.safetensors
# 预期：77e9e088e7801d3196c6a3fff057b06353a2b75e4e1a4e9d99b3de41e5f230ab
.venv/bin/python deai.py --adapter-path template \
  --input /path/to/draft.txt --run-dir runs/example
```

结果在 `runs/example/result.txt`。示例附件只包含微调过的 decoder block，**还需上述匹配的基础模型**；不要把它当完整 1.1B 模型。

## 流程 B：用自己的风格训练

```bash
.venv/bin/python deai.py \
  --style /path/to/licensed-style.txt \
  --input /path/to/draft.txt \
  --run-dir runs/my-style
```

结果在 `runs/my-style/result.txt`。运行目录还包含训练/验证数据、配置、统计报告及 `weights/`。目标风格文本需至少 12 个不同训练单元；训练和验证各需达到 15% 的实质增强覆盖率。可先加 `--prepare-only` 检查数据，也可加 `--steps 80` 指定训练步数（默认按样本数选 20–200 步）；每次须使用**不存在的新运行目录**。

数据增强具体实现见 [`style_data.py`](style_data.py)。没有安全变换的单元只使用弱的“去模板”增强；`report.json` 分开记录实质覆盖率及弱增强数。弱增强占比过高时，模型可能只学会删套话，不代表真正学到目标风格。请使用多段、多样化且有使用权的英文文本，并以独立人工改写样本检验质量。

两种流程均可添加 `--spelling-rate 0.01 --grammar-rate 0.01` 开启输出层错误控制（分别取值 0–0.05，默认均为 0）；这会降低文本质量，不建议用于正式用途。`--seed` 控制数据拆分和抽样。
