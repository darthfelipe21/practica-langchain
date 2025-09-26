from typing import Annotated  # Agregar informacion adicional a los tipos de datos
from typing_extensions import TypedDict  # Esquema de estados del chatbot
from langgraph.graph import StateGraph, START, END  # Construit un grafo de nodos
from langgraph.graph.message import add_messages  # Faciliatr lista de mensajes de estado
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv()

# Definimos el esatdo del chatbot
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Inicializamos el modelo LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

# Funcion principal del chatbot
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Construccion del grafo
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Añadir memoria de persistencia
memory = MemorySaver()

# Compilar el grafo
graph = graph_builder.compile(checkpointer=memory)


# Crear el chatbot
def chat_with_memory(user_input, thread_id="user-1"):
    """Chat con memoria persistente"""
    config = {"configurable":{"thread_id":thread_id}}

    result = graph.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )

    return result["messages"][-1].content