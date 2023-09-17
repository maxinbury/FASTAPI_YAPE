import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory

      
router = APIRouter()  
  

class Configuration(BaseModel):
    AZURE_SEARCH_SERVICE_NAME: str = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
    AZURE_SEARCH_ADMIN_KEY: str = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    AZURE_SEARCH_VECTOR_INDEX_NAME:str = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
    AZURE_SEARCH_API_KEY:str = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
    MONGODB_CONNECTION_STRING:str =os.getenv("MONGODB_CONNECTION_STRING")
    MONGODB_SESSION:str =os.getenv("MONGODB_SESSION")
class Query(BaseModel):
    question: str



def load_chain(config: Configuration):
    history_chat = MongoDBChatMessageHistory(
    connection_string=config.MONGODB_CONNECTION_STRING,
    session_id=config.MONGODB_SESSION
    )

    prompt_template = """ Eres un agente de soporte de servicio que está hablando con un cliente que te dara su nombre y tu lo recordaras para contestar las preguntas. Utiliza únicamente el historial del chat y la siguiente información {context}
                          para responder de manera útil a la pregunta. Si no conoces la respuesta - indica que no la conoces. Mantén tus respuestas breves, compasivas e informativas,
                          escucha activa, resolucion de problemas, manejo de objeciones y orientacion al cliente.
                          
                          Además, debes tener en cuenta estos factores adicionales en tus interacciones:
                          1. De vez en cuando, incentiva a los usuarios para que dejen comentarios sobre la plataforma, pero hazlo de manera equilibrada para no resultar insistente o invasivo. Usa tu juicio para determinar el momento apropiado para esta solicitud.
                          2. En caso de que no puedas responder una consulta, proporciona al usuario un hipervínculo a nuestro número de WhatsApp, donde un miembro de nuestro equipo podrá asistirle de manera más completa. Este número se puede encontrar en el historial del chat.
                          3. La seguridad de nuestros usuarios es fundamental. Si un usuario sin querer revela datos sensibles, como un número de tarjeta de crédito, DNI, numero de cuenta, recuérdales amablemente sobre la importancia de la protección de sus datos personales y asegúrales que nuestra plataforma nunca solicitará dicha información de manera directa.
                Ejemplo:
            Pregunta: me llamo maxi y necesito crear una cuenta?
            Respuesta: Hola Maxi! Encantado de conocerte. Para crear una cuenta en Yape, necesitarás seguir estos pasos:
                        Descarga la aplicación Yape desde la tienda de aplicaciones.
                        Abre la aplicación y toca en el botón "Crear Cuenta".
                        Registra tu número de celular y toca en "Continuar".
                        Recibirás un código de confirmación por mensaje de texto. Ingrésalo antes de que se acabe el tiempo y toca en "Validar código".
                        Llena los campos solicitados, como tu DNI y correo electrónico.
                        Crea una clave de 6 dígitos y selecciona el método de registro. Si tienes una tarjeta del BCP u otras tarjetas VISA compatibles, podrás registrarla.
                        Sigue los pasos para validar tu identidad y proporciona los datos de tu tarjeta. ¡Y listo! Ya tendrás tu cuenta en Yape y estarás listo para empezar a yapear.
                        Si tienes alguna otra pregunta o necesitas más ayuda, no dudes en decírmelo. Estoy aquí para ayudarte, Maxi.
            Pregunta: puedo usar yape desde argentina?
            Respuesta: Maxi! Lamentablemente, Yape solo está disponible para su uso dentro de Perú. Actualmente, no es posible utilizar Yape desde Argentina u otros países. Pero no te preocupes, si tienes alguna otra pregunta o necesitas ayuda con algo más, estaré encantado de ayudarte en lo que pueda. ¡Estoy aquí para ayudarte, Maxi!
    Question: {question}
    Answer here:"""  
  
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    ) 

    # Creating instances of AzureCognitiveSearchRetriever and ChatOpenAI
    retriever = AzureCognitiveSearchRetriever(service_name=config.AZURE_SEARCH_SERVICE_NAME, api_key=config.AZURE_SEARCH_ADMIN_KEY, index_name=config.AZURE_SEARCH_VECTOR_INDEX_NAME, content_key="content", top_k=10)
    llm=ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-16k-0613", openai_api_key=config.OPENAI_API_KEY)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=ConversationBufferMemory(memory_key="chat_history"), combine_docs_chain_kwargs={"prompt": PROMPT})
    return chain, history_chat

@router.post("/messages")  
async def query_chatbot(query: Query, config: Configuration = Depends(Configuration)):
    try:
        chain, history_chat = load_chain(config)
        output = chain.run(question=query.question, chat_history=[])

        # Agrega el mensaje del usuario a la base de datos MongoDB.
        history_chat.add_user_message(query.question)
        # Agrega la respuesta de la IA a la base de datos MongoDB.
        history_chat.add_ai_message(output)

        return {"result": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))