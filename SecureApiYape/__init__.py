import logging  
from fastapi import FastAPI  
from routers import chatbot  
import azure.functions as func  
  
app = FastAPI()  
app.include_router(chatbot.router, prefix="/api")  
  
def main(req: func.HttpRequest, context:func.Context) -> func.HttpResponse:  
    return func.AsgiMiddleware(app).handle(req, context)  
