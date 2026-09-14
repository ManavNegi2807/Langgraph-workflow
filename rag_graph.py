

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DOCS = [
    Document(page_content="COCOMO estimates effort as E = a(KLOC)^b person-months.", metadata={"source": "cocomo"}),
    Document(page_content="Function points size software from inputs, outputs, and files.", metadata={"source": "fp"}),
]


class State(TypedDict):
    question: str
    documents: list[Document]
    answer: str


def build_retriever(docs: list[Document]):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings())
    return store.as_retriever(search_kwargs={"k": 2})


retriever = build_retriever(DOCS)


def retrieve(state: State) -> State:
    return {"documents": retriever.invoke(state["question"])}


def generate(state: State) -> State:
    context = "\n".join(d.page_content for d in state["documents"])
    prompt = f"Answer using this context:\n{context}\n\nQuestion: {state['question']}"
    return {"answer": llm.invoke(prompt).content}


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
graph = g.compile()


if __name__ == "__main__":
    question = "What is COCOMO?"
    result = graph.invoke({"question": question})
    print(result["answer"])
