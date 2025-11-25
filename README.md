<h1 align="center"><strong>⚕️ 多智能体医疗助手 :<h6 align="center">基于AI的多智能体系统，用于医疗诊断和辅助</h6></strong></h1>

<!-- ![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch) -->

![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph - Version](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge&logo=langgraph)
![LangChain - Version](https://img.shields.io/badge/LangChain-0.3+-teal?style=for-the-badge&logo=langchain)
![Qdrant Client - Version](https://img.shields.io/badge/Qdrant-1.13+-red?style=for-the-badge&logo=qdrant)
![Pydantic - Version](https://img.shields.io/badge/Pydantic-2.10+-red?style=for-the-badge&logo=pydantic)
![FastAPI - Version](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
![Docling - Version](https://img.shields.io/badge/Docling-3.1+-orange?style=for-the-badge&logo=docling)
[![Generic badge](https://img.shields.io/badge/License-Apache-<COLOR>.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Multi-Agent-Medical-Assistant.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)

## 📚 目录

- [概述](#概述)
- [演示](#演示)
- [技术流程图](#技术流程图)
- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [安装和设置](#安装)
- [使用方法](#使用方法)
- [贡献](#贡献)
- [许可证](#许可证)
- [引用](#引用)
- [联系](#联系)

---

## 📌 概述

**多智能体医疗助手** 是一个 **基于AI的聊天机器人**，旨在协助 **医疗诊断、研究和患者互动**。

🚀 **基于多智能体智能**，该系统集成了：

- **🤖 大型语言模型 (LLMs)**
- **🖼️ 计算机视觉模型** 用于医学图像分析
- **📚 检索增强生成 (RAG)** 利用向量数据库
- **🌐 实时网络搜索** 获取最新的医学见解
- **👨‍⚕️ 人机验证** 验证基于AI的医学图像诊断

---

## 🛡️ 技术流程图

![Technical Flow Chart](assets/final_medical_assistant_flowchart_light_rounded.png)

---

## ✨ 核心功能

- 🤖 **多智能体架构** : 专门的智能体协同工作，处理诊断、信息检索、推理等任务
- 🔍 **高级智能体RAG检索系统** :

  - 基于Docling的解析，从PDF中提取文本、表格和图像。
  - 嵌入markdown格式的文本、表格和基于LLM的图像摘要。
  - 基于LLM的语义分块，具有结构边界意识。
  - 基于LLM的查询扩展，包含相关的医学领域术语。
  - Qdrant混合搜索，结合BM25稀疏关键字搜索和密集嵌入向量搜索。
  - 基于HuggingFace Cross-Encoder的检索文档块重排序，以获得准确的LLM响应。
  - 输入输出防护，确保安全和相关的响应。
  - 在响应中提供源文档和图像的链接。
  - 基于置信度的智能体间切换，防止幻觉。
- 🏥 **医学影像分析**

  - 脑肿瘤检测 (待实现)
  - 胸部X光疾病分类
  - 皮肤病变分割
- 🏥 **医院信息查询系统** :

  - 医院基础信息查询（地址、联系方式、科室等）
  - 主治医师信息查询
  - 医生预约信息查询
  - 医院服务和设施信息
- 🌐 **实时研究整合** : 网络搜索智能体，检索最新的医学研究论文和发现
- 📊 **基于置信度的验证** : 对数概率分析确保医学推荐的高准确性
- 🎙️ **语音交互功能** : 无缝的语音到文本和文本到语音，由Eleven Labs API提供支持
- 👩‍⚕️ **专家监督系统** : 医疗专业人员在最终输出前验证AI结果
- ⚔️ **输入和输出防护** : 确保安全、无偏见和可靠的医学响应，同时过滤有害或误导性内容
- 💻 **直观的用户界面** : 为医疗专业人员设计，技术门槛低

---

## 🛠️ 技术栈

| 组件                   | 技术                            |
| ---------------------- | ------------------------------- |
| 🔹**后端框架**   | FastAPI                         |
| 🔹**智能体编排** | LangGraph                       |
| 🔹**文档解析**   | Docling                         |
| 🔹**知识存储**   | Qdrant 向量数据库               |
| 🔹**医学影像**   | 计算机视觉模型                  |
|                        | • 脑肿瘤: 目标检测 (PyTorch)   |
|                        | • 胸部X光: 图像分类 (PyTorch)  |
|                        | • 皮肤病变: 语义分割 (PyTorch) |
| 🔹**防护机制**   | LangChain                       |
| 🔹**语音处理**   | Eleven Labs API                 |
| 🔹**前端**       | HTML, CSS, JavaScript           |
| 🔹**部署**       | Docker, GitHub Actions CI/CD    |

---

### 安装

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git  
cd Multi-Agent-Medical-Assistant  
```

### 2️⃣ 创建并激活虚拟环境

- 如果使用conda:

```bash
conda create --name <environment-name> python=3.11
conda activate <environment-name>
```

- 如果使用python venv:

```bash
python -m venv <environment-name>
source <environment-name>/bin/activate  # 对于Mac/Linux
<environment-name>\Scripts\activate     # 对于Windows  
```

### 3️⃣ 安装依赖

语音服务需要ffmpeg。

- 如果使用conda:

```bash
conda install -c conda-forge ffmpeg
```

```bash
pip install -r requirements.txt  
```

- 如果使用python venv:

```bash
winget install ffmpeg
```

```bash
pip install -r requirements.txt  
```

### 4️⃣ 设置API密钥

- 创建一个 `.env` 文件并添加所需API密钥。

```yaml
# LLM Configuration (Azure Open AI - gpt-4o used in development)
# If using any other LLM API key or local LLM, appropriate code modification is required
deployment_name= 
model_name=gpt-4o
azure_endpoint=
openai_api_key=
openai_api_version=

# Embedding Model Configuration (Azure Open AI - text-embedding-ada-002 used in development)
# If using any other embedding model, appropriate code modification is required
embedding_deployment_name=
embedding_model_name=text-embedding-ada-002
embedding_azure_endpoint=
embedding_openai_api_key=
embedding_openai_api_version=

# Speech API Key (Free credits available with new Eleven Labs Account)
ELEVEN_LABS_API_KEY=

# Web Search API Key (Free credits available with new Tavily Account)
TAVILY_API_KEY=

# Hugging Face Token - using reranker model "ms-marco-TinyBERT-L-6"
HUGGINGFACE_TOKEN=

# (OPTIONAL) If using Qdrant server version, local does not require API key
QDRANT_URL=
QDRANT_API_KEY=
```

### 5️⃣ 运行应用程序

- 在激活的环境中运行以下命令。

```bash
python app.py
```

应用程序将在以下地址可用: [http://localhost:8000](http://localhost:8000)

### 6️⃣ 向向量数据库注入额外数据

根据需要运行以下命令之一。

- 一次注入一个文档:

```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- 从目录注入多个文档:

```bash
python ingest_rag_data.py --dir ./data/raw
```
