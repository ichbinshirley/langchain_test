# 构建一个分析候选人简历的输出解析器， 可以生成对简历的分析，冰上接受用户的提问

import os
openai_api_key = os.getenv('OPENAI_API_KEY')


from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser
)
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader


# 构建prompt
# 第一个：输出简历分析
template_one = '''以下是一段关于面试候选人自我介绍的文本，\
需要按照一定的格式生成对候选人个人情况的分析,\
{context}
并按照要求的格式进行输出。
{format}
'''
first_prompt = ChatPromptTemplate.from_template(template=template_one)

# 第二个：回答用户问题
template_two = '''以下是一段关于面试候选人自我介绍的文本，\
需要按照一定的格式生成对候选人个人情况的分析,\
{context}
并回答用户的问题。
{question}
'''
second_prompt = ChatPromptTemplate.from_template(template=template_two)

# 加载简历pdf文本
text = ''
loader = PyPDFLoader('/Users/shirley/Documents/Apps/chroma/王梅雪中文简历_test.pdf')
pages = loader.load()
for i in range(len(pages)):
    text += pages[i].page_content

# 构建简历分析框架
YOE_schema = ResponseSchema(name='工作年限',
                            description='候选人拥有多少年的工作经验，\
                            输出结果类型应为整数，\
                            如果没有找到该信息，则回答None')
industry_schema = ResponseSchema(name='能力',
                            description='提取有关候选人能力描述的句子,\
                            如果没有找到该信息，则回答None')
dataskill_schema = ResponseSchema(name='行业经验',
                            description='候选人曾在哪些行业工作过，例如电商,\
                            如果没有找到该信息，则回答None')
analysis_schema = [YOE_schema, industry_schema, dataskill_schema]
output_parser = StructuredOutputParser.from_response_schemas(analysis_schema)
format_instructions = output_parser.get_format_instructions('True')

# 调用llm
user_input = '这个候选人有过几段工作经历，分别是在哪个公司'
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
message_one = first_prompt.format_messages(context=text,
format=format_instructions)
response1=llm(message_one)
print(response1)
message_two = second_prompt.format_messages(context=text,
question=user_input)
response2=llm(message_two)
print(response2)



# # 构建llm chain
# llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
# chain_one = LLMChain(
#     llm=llm,
#     prompt=first_prompt,
#     output_parser=output_parser
# )
# # chain_two = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     retriever=vectorstore.as_retriever(),
# #     chain_type_kwargs={'prompt': second_prompt},
# # )
# # overall_chain = SequentialChain(
# #     chains=[chain_one, chain_two]
# # )
# #
# # # 运行chain
# # user_question = '该候选人在最近两份工作的主要工作职责是什么？'
# # print(overall_chain.run({'query': user_question}))
#
# print(chain_one.run(text))