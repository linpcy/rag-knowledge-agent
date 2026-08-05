import streamlit as st
from rag_system import RAGSystem

st.set_page_config(page_title="知识库问答", page_icon="📚")
st.title("📚 个人知识库问答 Agent")
st.markdown("基于 LangChain + Ollama 的本地 RAG 系统")

@st.cache_resource
def get_rag_system():
    return RAGSystem()

try:
    rag = get_rag_system()
    question = st.text_input("请输入你的问题：", placeholder="例如：这份文档的主要内容是什么？")

    if st.button("提问", type="primary"):
        if question.strip():
            with st.spinner("正在思考中..."):
                result = rag.answer(question)
            st.markdown("### 回答")
            st.write(result["answer"])
            with st.expander("查看引用来源"):
                for i, source in enumerate(result["sources"], 1):
                    st.markdown(f"**来源 {i}：**")
                    st.text(source)
        else:
            st.warning("请输入问题后再点击提问！")

except Exception as e:
    st.error(f"系统初始化失败：{str(e)}")
    st.info("请确保：1. Ollama已启动 2. 模型已下载 3. data目录下有PDF文件")