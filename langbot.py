
from langgraph.graph import  StateGraph , START,END
from langchain_openai import OpenAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
load_dotenv()


class LangBot(TypedDict) :
    message : Annotated[str, "Then message to send to the bot"]
    answer: Annotated[str, "Then answer from the bot"]  

def llmq(state: LangBot) -> LangBot:

    message  = state['message']
    prompt = f'Answer the following question {message}'
    answer = llm.invoke(prompt).content

    state['answer'] = answer

    return state

llm = OpenAI(model_name="gpt-4", temperature=0.7, max_tokens=2000)

graph = StateGraph(LangBot)

graph.add_node("llmq",llmq)
graph.add_edge(START,"llmq")
graph.add_edge("llmq",END)

result =graph.compile()
intial_state = {'message': 'How far is moon from the earth?'}

final_state = result.invoke(intial_state)
