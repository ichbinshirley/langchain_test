import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 加载文档
loader = PyPDFLoader('/Users/shirley/Documents/Apps/chroma/绩效管理的定义.pdf')
pages = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                               chunk_overlap=100)
split_docs = text_splitter.split_documents(pages)

# 向量化
embeddings = OpenAIEmbeddings()

# 存储到向量数据库chroma
vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name='langchain_test1')

# 构建prompt
teamplate = '''
使用以下上下文来回答用户的问题。如果你不知道，只需回答不知道，不要试图编造答案。尽量简要地回答。
{context}
问题: {question}
有用的回答：
'''
chat_prompt = PromptTemplate.from_template(template)

# 构建qa chain，回调
handler = StdOutCallbackHandler()
llm = ChatOpenAI(temperature=0, max_tokens=500)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={'prompt': chat_prompt},
    callbacks=[handler]
)

# 输入问题
question = '绩效管理的作用是什么'

# 调用llm得到输出
result = qa_chain({"query": question})
print(result["result"])