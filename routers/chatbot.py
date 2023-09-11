import os
from fastapi import APIRouter, HTTPException 
from pydantic import BaseModel
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory    
      
router = APIRouter()  
  
class Configuration(BaseModel):
    AZURE_SEARCH_SERVICE_NAME: str = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
    AZURE_SEARCH_ADMIN_KEY: str = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    AZURE_SEARCH_VECTOR_INDEX_NAME:str = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
    AZURE_SEARCH_API_KEY:str = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
    question: str

def load_chain(config: Configuration):
    memory = ConversationBufferMemory(  
        memory_key="chat_history", return_messages=True, output_key="answer"  
    )    
    prompt_template = """ Eres un agente de soporte de servicio que está hablando con un cliente. Utiliza únicamente el historial del chat y la siguiente información {context}
                          para responder de manera útil a la pregunta. Si no conoces la respuesta - indica que no la conoces. Mantén tus respuestas breves, compasivas e informativas, escucha activa, resolucion de problemas, manejo de objeciones y orientacion al cliente.

    Question: {question}
    Answer here:"""  
  
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    ) 

    # Creating instances of AzureCognitiveSearchRetriever and ChatOpenAI
    retriever = AzureCognitiveSearchRetriever(service_name=config.AZURE_SEARCH_SERVICE_NAME, api_key=config.AZURE_SEARCH_ADMIN_KEY, index_name=config.AZURE_SEARCH_VECTOR_INDEX_NAME, content_key="content", top_k=10)
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key=config.OPENAI_API_KEY)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT})
    return chain

history = []
@router.post("/query")  
async def query_chatbot(config: Configuration):
    global history
    try:
        chain = load_chain(config)
        output = chain.run(question=config.question, chat_history=[])
        history.append((config.question, output))

        return {"result": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))