---
language:
- zh
- en
license: apache-2.0
base_model:
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-7B
tags:
- legal
- law
- chinese
- reasoning
- legal-ai
- legal-reasoning
- qwen
---

# LegalOne: A Family of Foundation Models for Reliable Legal Reasoning

<p align="center">
  <img src="fig/logo.png" alt="LegalOne Logo" width="20%"/>
</p>

<p align="center">
  <a href="https://github.com/CSHaitao/LegalOne" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-LegalOne-blue?style=flat-square&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  <a href="https://github.com/DavidMiao1127/LegalKit" target="_blank">
    <img src="https://img.shields.io/badge/Evaluation-LegalKit-purple?style=flat-square&logo=github&logoColor=white" alt="LegalKit"/>
  </a>
</p>

## 概述

近年来，法律领域对可靠AI系统的需求快速增长。然而，法律推理既**知识密集**又**结构密集**，通用LLM往往存在法律知识理解不足、推理与实践脱节等问题，难以满足法律系统对可靠性的要求。

**LegalOne** 是一系列专门为中文法律领域训练的LLM，采用**多阶段训练框架**来联合增强法律知识和推理能力。模型基于 **昇腾 Atlas 910B** 计算平台与 **昇思 MindSpore** AI 框架完成训练。

### 训练方法

- **中期训练**：基于困惑度的数据调度方法-Plasticity-Adjusted Sampling (PAS)，从广泛、异构的通用数据平滑过渡到专业化法律任务，在有效注入法律知识的同时避免灾难性遗忘。

- **监督微调**：我们建立了一个模拟专业法律工作流程的代理系统Legal Agentic CoT Distillation (LEAD)，能够综合大规模、高一致性的推理轨迹，培养模型执行可靠推理的能力。

- **强化学习**：采用多阶段课程学习，从简单到复杂逐步塑造推理能力，形成更内化、更自主的"法律思维"模式。

### 模型性能

**LegalOne-8B** 在**法规解释、判例法推理、法律问答、文档起草**等任务上超越通用 LLM 和现有法律模型。在 **LexEval、JecQA** 等权威评测中，整体性能可媲美更大参数规模的通用模型（如 **DeepSeek-R1、Qwen3-Max**），并在部分任务上实现超越。尤其在**法律概念理解、法条记忆、多跳推理**等关键任务上，**LegalOne-8B** 达到了当前开源模型的**领先水平**。

<p align="center">
  <img src="fig/overview.png" alt="LegalOne Overview" width="90%"/>
</p>

---

## 模型系列

本次发布包含 1.7B、4B 和 8B 三个参数规模的模型，覆盖从轻量级部署到高性能应用的不同场景需求。

| 模型 | 参数量 | 基座模型 | 支持语言 | 链接 |
|-------|-----------|------------|---------------------|------|
| LegalOne-1.7B | 1.7B | Qwen3-1.7B-Base | 中文 & 英文 | [🤗 HF Link](https://huggingface.co/CSHaitao/LegalOne-1.7B) |
| LegalOne-4B | 4B | Qwen3-4B-Base | 中文 & 英文 | [🤗 HF Link](https://huggingface.co/CSHaitao/LegalOne-4B) |
| LegalOne-8B | 8B | Qwen3-8B-Base | 中文 & 英文 | [🤗 HF Link](https://huggingface.co/CSHaitao/LegalOne-8B) |

---

## 使用方法

LegalOne 可以像普通的 Qwen3 模型一样使用。你可以使用 [vLLM](https://github.com/vllm-project/vllm) 或 [Sglang](https://github.com/sgl-project/sglang) 等工具进行部署，也可以直接使用 transformers 进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "CSHaitao/LegalOne-8B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("CSHaitao/LegalOne-8B")

input_text = "请根据以下提供的案件事实，从法律角度进行分析，并预测法院可能作出的判决。"
messages = [{"role": "user", "content": input_text}]

inputs = tokenizer(
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ),
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 输出格式

LegalOne 采用"先思考后回答"的方式，输出格式如下：

```
思考过程
...
[最终回答]
...
```

---

## 数据说明

### 中期训练语料

LegalOne 的中期训练采用精心构建的混合语料库，整合通用数据、法律数据和合成数据三大类，总计约 **100B tokens**，为模型提供扎实的知识基础。

- **通用数据**：整合 FinWeb-Edu、FinePDFs、FineWiki、SkyPile-150B、IndustryCorpus、OpenNewsArchive、MathPile、Wanjuan、BaiduBaike-5.63M 等高质量开源语料，在法律专业化训练中有效缓解灾难性遗忘，保持模型的通用语言理解能力。

- **法律数据**：涵盖学术论文、法律评论、法律注释、法律咨询、法院判决、司法意见、法律教材、法律百科、指导意见、法律法规等多维度法律文本，严格筛选近5年的法律文档，仅保留现行有效的法规条文。

- **合成数据**：依托 Qwen3-255B-A22B-Thinking 从权威法律源生成专家级、多角度推理样本，并通过精心设计的提示词生成多种写作风格和视角的忠实改写，有效提升每个 token 的学习信号密度。

### 监督微调数据

为系统性地获得高质量的法律思维链数据，我们开发了模拟专业法律工作流程的 **LEAD (Legal Agentic CoT Distillation)** 系统。该流程包含四个阶段：

1. **提示词收集**：通过结构化案例语料库构建、结构逻辑蒸馏、多视角用户模拟和真实查询对齐四个环节，构建高质量的多样化提示词模板

2. **代理式思维链综合**：模拟法律专家真实认知过程的框架，将抽象的司法推理形式化为结构化的、明确的代理工作流

3. **轨迹精炼**：通过知识内化消除教师-学生模型间的信息差距，通过推理收敛将多阶段局部思维链合并为全局连贯的端到端推理轨迹

4. **质量控制**：采用启发式过滤和 LLM-as-judge 评估，从推理质量、一致性、答案-推理对齐、简洁性、语言等多个维度确保生成内容的可靠性和实用性

基于高质量法律文书，通过智能体 CoT 蒸馏管道，我们合成了涵盖法律咨询、判决预测、法律摘要、法律适用、文书生成等多种经典司法场景的数据。结合部分开源的通用指令遵循数据以保持模型的通用能力，最终获得 **500k** 高质量监督微调数据。

### 强化学习数据

我们收集并合成了一批适合法律领域进行强化学习的任务，涵盖从基础记忆到高级推理的完整能力谱系。这些强化学习任务具有可验证性，能够提供明确的奖励信号，指导模型通过强化学习逐步提升法律推理能力。

---

## 评测

为确保评测结果的可复现性与透明度，我们推出了 **[LegalKit](https://github.com/DavidMiao1127/LegalKit)** 评测工具包，所有实验结果均基于 LegalKit 评测得出。

**LegalKit** 是一个实用且可扩展的法律领域大模型评测工具包，统一了以下流程：数据集适配、模型生成、离线 JSON 评测、LLM-as-Judge 评审，同时提供可选的轻量级 Web UI，方便非命令行用户操作。

LegalOne 系列模型在法律基础能力上表现突出，在 LexEval、JecQA 等测试集上，整体表现接近甚至超越参数规模显著更大的通用模型。

**说明**：我们同时意识到现在的评估数据集没有聚焦于法律实务现实场景的考察，我们欢迎大家对模型进行更详细的评测并反馈 bad case 和 good case，我们会进一步改进。我们同时也在推出基于rubric评测的实务数据集，敬请期待！

---

## 引用

如果你觉得这项工作有用，请引用：

```bibtex
@misc{legalone-2025,
  title={LegalOne: A Family of Foundation Models for Reliable Legal Reasoning},
  author={LegalOne Team},
  year={2025},
  url={https://github.com/CSHaitao/LegalOne}
}
```

---

## 许可证

本项目采用 **Apache 2.0 许可证** - 详见 [LICENSE](https://github.com/CSHaitao/LegalOne/blob/main/LICENSE) 文件。

---

## 免责声明

**LegalOne** 是基于深度学习技术构建的法律大语言模型，旨在为法律研究和应用提供辅助工具。模型可以提供有价值的法律信息分析和推理参考，但**不应视为法律专业人士的替代品**。

在重要的法律事务中，建议您咨询专业的法律顾问或律师。模型输出仅供参考，不构成正式的法律意见或建议。

---

## 联系方式

如果您对 LegalOne 有任何疑问、建议或想法，欢迎访问我们的 **[GitHub 仓库](https://github.com/CSHaitao/LegalOne)** 提交 Issue 或参与讨论。

更多详细信息请访问：**https://github.com/CSHaitao/LegalOne**
