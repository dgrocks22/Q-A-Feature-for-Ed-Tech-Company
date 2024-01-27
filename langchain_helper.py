import os
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import chardet
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
llm= GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.1)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"
def create_vector_db():
# Detect encoding
    with open("codebasics_faqs.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read())
    # Use detected encoding
    systumm = CSVLoader(file_path="codebasics_faqs.csv", source_column='prompt', encoding=result['encoding'])
    data=systumm.load()
    vector_db=FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vector_db.savelocal(vectordb_file_path)
def get_qa_chain():
    vector_db=FAISS.from_documents(vectordb_file_path,embedding=instructor_embeddings)
    retriever = vector_db.as_retriever(score_threshold = 0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    return chain
if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you provide job assistance and also do you provide job gurantee?"))





