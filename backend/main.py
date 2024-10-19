from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model import chat_with_llm
app = FastAPI()

# Enable CORS
origins = ['http://localhost:3000']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# In-memory storage for messages (replace with a database in production)
messages = []

class Message(BaseModel):
    content: str
    sender: str


# Input model to receive a string
class ModelInput(BaseModel):
    input_text: str
    user_id : str

# Route to accept input and process it with the ML model
@app.post("/api/predict")
async def predict(input_data: ModelInput):
    input_str = input_data.input_text
    user_id = input_data.user_id  # Extract the input string from the request
    if not input_str:
        raise HTTPException(status_code=400, detail="Input string is empty")

    # Pass the input to your ML model and get a result
    result = chat_with_llm(user_id,input_str)

    # Return the result to the client
    return {"input": input_str, "result": result}


# @app.post("/api/send_message")
# async def send_message(message: Message):
#     messages.append(message)
#     # Here you can add your logic to generate a response
#     response = Message(content=f"Server received: {message.content}", sender="Server")
#     messages.append(response)
#     return {"status": "success", "message": "Message sent", "response": response}

# @app.get("/api/get_messages")
# async def get_messages():
#     return f"Fuck is wrong with you"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = Message(content=data, sender="User")
            messages.append(message)
            # Generate and send response
            response = Message(content=f"Server received: {data}", sender="Server")
            messages.append(response)
            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        print("WebSocket disconnected")