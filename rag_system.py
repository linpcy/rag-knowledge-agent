import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA


class RAGSystem:
    def __init__(self, data_dir="data", db_dir="chroma_db", model_name="qwen2.5:7b"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.model_name = model_name

        self.embeddings = OllamaEmbeddings(model=model_name)
        self.llm = OllamaLLM(model=model_name)

        self.vectorstore = self._load_or_create_vectorstore()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def _load_or_create_vectorstore(self):
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            print("加载已有向量库...")
            return Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )

        print("创建新的向量库...")
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.data_dir, filename)
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())

        if not documents:
            raise ValueError(f"在 {self.data_dir} 目录下没有找到PDF文件！")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        return vectorstore

    def answer(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }