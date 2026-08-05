# rag-knowledge-agent

基于 LangChain + Ollama 的本地知识库问答系统。

## 功能
- 📄 PDF 文档智能解析与向量化
- 🤖 本地大模型离线推理（零 API 费用）
- 💬 Streamlit 交互式 Web 界面
- 🔍 检索增强生成（RAG），答案可溯源

## 技术栈
Python | LangChain | Ollama | Chroma | Streamlit

## 快速开始

### 1. 安装 Ollama 并下载模型
```bash
ollama pull qwen2.5:7b