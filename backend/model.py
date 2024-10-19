from langchain_google_genai import ChatGoogleGenerativeAI
import os

from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.utilities import WikipediaAPIWrapper

os.environ["LANGCHAIN_TRACING_V2"] = "true"
#_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Plan-and-execute"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0'

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

from langchain_community.tools.tavily_search import TavilySearchResults


ddg_search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [TavilySearchResults(max_results=3),ddg_search,wikipedia]


prompt = (
    f"You are a human study buddy. The user will provide you with sets of commands, either to solve questions for them or to generate questions. If provided with questions, answer comprehensively and simplify explanations as much as possible, assuming no prior knowledge from the user. If asked to generate questions, cover all relevant topics in the provided texts by giving the user as many questions as possible. If you don't have an answer to a particular question, use the available tools ({tools}) to search online for the latest information. Properly format your responses with backticks or LaTeX when necessary. Give multiple and varied answers, making your responses as comprehensive as possible also try to be jovial as much as possible be much concerned about the user well being remember you are a study buddy and a companion as well as a friend to the user. If the user's prompt is ambiguous or unclear, request clarification or provide a range of possible interpretations. Incorporate user feedback to improve your performance over time."
)

# We can add "chat memory" to the graph with LangGraph's checkpointer
# to retain the chat context between interactions

memory = MemorySaver()

user_memories = {}

def get_or_create_user_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = MemorySaver()
    return user_memories[user_id]

def create_user_graph(user_id):
    user_memory = get_or_create_user_memory(user_id)
    return create_react_agent(llm, tools=tools, checkpointer=user_memory, state_modifier=prompt)

graph = create_react_agent(llm, tools=tools, checkpointer=memory,state_modifier=prompt)
config = {"configurable": {"thread_id": "1"}}
def stream_response(stream):
    response = [x["messages"] for x in stream]
    return response[-1][-1].content

def chat_with_llm(user_id: str, input_value: str) -> str:
    user_graph = create_user_graph(user_id)
    inputs = {"messages": [("user", input_value)]}
    config = {"configurable": {"thread_id": user_id}}
    return stream_response(user_graph.stream(inputs, config=config, stream_mode="values"))