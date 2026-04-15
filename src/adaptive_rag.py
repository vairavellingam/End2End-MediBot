import os
from typing import List
from typing import Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START

from src.prompt import (
    system_prompt,
    router_system_prompt,
    retrieval_grader_prompt,
    hallucination_grader_prompt,
    answer_grader_prompt,
    query_rewriter_prompt,
    rag_generation_prompt,
)


# LLM

def _get_llm():
   
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)



# Pydantic schemas for structured outputs

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Route to 'vectorstore' for medical knowledge base queries, "
                    "'web_search' for recent events or out-of-scope questions."
    )


class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""
    binary_score: str = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )


class GradeHallucinations(BaseModel):
    """Binary score — is the generation grounded in the retrieved facts?"""
    binary_score: str = Field(
        description="'yes' if grounded in facts, 'no' if it contains hallucinations."
    )


class GradeAnswer(BaseModel):
    """Binary score — does the generation actually answer the question?"""
    binary_score: str = Field(
        description="'yes' if the answer resolves the question, 'no' otherwise."
    )


# Graph state 

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    retries: int  


# Chain / tool builders  

def build_adaptive_rag(retriever):
    """
    Build and compile the Adaptive RAG LangGraph app.

    Args:
        retriever: A LangChain retriever backed by Pinecone (from app.py).

    Returns:
        Compiled LangGraph StateGraph app.
    """
    llm = _get_llm()

    # -- Router --
    structured_router = llm.with_structured_output(RouteQuery)
    question_router = router_system_prompt | structured_router

    # -- Retrieval grader --
    structured_doc_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = retrieval_grader_prompt | structured_doc_grader

    # -- RAG generation chain --
    rag_chain = rag_generation_prompt | llm | StrOutputParser()

    # -- Hallucination grader --
    structured_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_grader = hallucination_grader_prompt | structured_hallucination_grader

    # -- Answer grader --
    structured_answer_grader = llm.with_structured_output(GradeAnswer)
    answer_grader = answer_grader_prompt | structured_answer_grader

    # -- Query rewriter --
    question_rewriter = query_rewriter_prompt | llm | StrOutputParser()

    # -- Web search tool --
    web_search_tool = TavilySearchResults(k=3)

    
    # Node functions
    
    def retrieve(state: GraphState) -> GraphState:
        print("---RETRIEVE FROM PINECONE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state: GraphState) -> GraphState:
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        retries = state.get("retries", 0) + 1
        generation = rag_chain.invoke({"context": documents, "question": question})
        print(f"  (attempt {retries})")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retries": retries,
        }

    def grade_documents(state: GraphState) -> GraphState:
        print("---GRADING DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for doc in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
           
            grade = (
                score.binary_score
                if hasattr(score, "binary_score")
                else score.get("binary_score", "no")
            )
            if grade == "yes":
                print("  [+] Relevant")
                filtered_docs.append(doc)
            else:
                print("  [-] Not relevant, filtered out")

        return {"documents": filtered_docs, "question": question}

    def transform_query(state: GraphState) -> GraphState:
        print("---REWRITING QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        print(f"  Rewritten: {better_question}")
        return {"documents": documents, "question": better_question}

    def web_search(state: GraphState) -> GraphState:
        print("---WEB SEARCH FALLBACK---")
        question = state["question"]
        results = web_search_tool.invoke({"query": question})
        web_content = "\n\n".join([r["content"] for r in results])
        web_doc = Document(page_content=web_content)
        return {"documents": [web_doc], "question": question}

    
    # Edge / conditional functions
   
    def route_question(state: GraphState) -> str:
        print("---ROUTING QUESTION---")
        question = state["question"]
        source = question_router.invoke({"question": question})
       
        datasource = (
            source.datasource
            if hasattr(source, "datasource")
            else source.get("datasource", "vectorstore")
        )
        if datasource == "web_search":
            print("  → Web search")
            return "web_search"
        print("  → Vectorstore")
        return "vectorstore"

    def decide_to_generate(state: GraphState) -> str:
        print("---DECIDING: GENERATE OR REWRITE?---")
        if not state["documents"]:
            print("  → No relevant docs found, rewriting query")
            return "transform_query"
        print("  → Relevant docs found, generating")
        return "generate"

    def grade_generation(state: GraphState) -> str:
        print("---GRADING GENERATION---")
        question   = state["question"]
        documents  = state["documents"]
        generation = state["generation"]
        retries    = state.get("retries", 0)

        
        if retries >= 2:
            print(f"  → Max retries ({retries}) reached, returning best answer")
            return "useful"

        # Convert Document objects to plain text string for the hallucination grader
        docs_text = "\n\n".join(doc.page_content for doc in documents)

        h_score = hallucination_grader.invoke({
            "documents": docs_text,  
            "generation": generation,
        })
        h_grade = (
            h_score.binary_score
            if hasattr(h_score, "binary_score")
            else h_score.get("binary_score", "yes")
        )

        if h_grade == "no":
            print(f"  → Hallucination detected (attempt {retries}), regenerating")
            return "not supported"

        a_score = answer_grader.invoke({"question": question, "generation": generation})
        a_grade = (
            a_score.binary_score
            if hasattr(a_score, "binary_score")
            else a_score.get("binary_score", "no")
        )

        if a_grade == "yes":
            print("  → Useful answer ✅")
            return "useful"

        print("  → Doesn't address question, rewriting")
        return "not useful"

   
    # Build the graph
    
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Entry point: route the question
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()
