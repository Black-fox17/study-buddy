from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0'
llm = ChatGoogleGenerativeAI(model = 'gemini-pro')
def chat_with_llm(input_value:str) -> str:
    result = llm.invoke(input_value)
    return result.content  