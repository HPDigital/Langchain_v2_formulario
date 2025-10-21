"""
Langchain_v2_formulario
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install python-docx langchain
#!pip install langchain faiss-cpu openai
#!pip install --upgrade --quiet  docx2txt
#!pip install pypdf


# In[2]:


# Lo que hace es carga todo los documentos, los separa en chuca, despeus cada chuk es vectorizado 
# y se gunarad en la base de datos, despues buca los chunks mas similares a la pregunta y esos los pasa como contexto
# de sta forma se ahorra en el costo de tokens y no se sobrepasa los limites
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain


def get_documents_from_doc(direc):
    loader = DirectoryLoader(direc, glob= ".\\*pdf",loader_cls=PyPDFLoader)
    docs = loader.load() 
    splitter = RecursiveCharacterTextSplitter(chunk_size =800, chunk_overlap = 80)
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embeddings)
    return vectorStore


def create_chain(vectorStore):
    llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.7, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True
                 )

    prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context} 
                                            Pregunta: {input}
                                            """)

    chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain



direc = "C:\\Users\\HP\\Desktop\\PROPUESTAS DE DIPLOMADO\\PROPUESTAS DE DIPLOMADO A POS-GRADO\\MODIFICADOS\\"
docs = get_documents_from_doc(direc)
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response = chain.invoke({ "input":"Cual es la justificacion de excel",
                        # "context" :docs --> al usar retrieval chain ya no se debe pasar contexto de esta forma
                        })

print(response["answer"])


# In[ ]:






if __name__ == "__main__":
    pass
