from typing import Dict, List, cast

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from pydantic import BaseModel, Field
from typing import List, TypedDict
import react_agent.prompts as prompts

import os

class AgentState(TypedDict):
    original_question: str
    questions: List[str]
    reasonings: List[str]
    documents: List[str]
    final_reasoning: str
    filter_documents: List[str]
    unfilter_documents: List[str]

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

endpoint_name = "recommendation_endpoint"
index_name = "rag.default.curso_processo_penal_epub_test_index"
embeddings_endpoint = os.getenv("ENDPOINT_MODEL")
host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")

vs_client = VectorSearchClient(disable_notice=True)
index = vs_client.get_index(endpoint_name=endpoint_name, index_name=index_name)

text_column = 'Conteudo'
        
columns = ["Conteudo", "Indice"]
        
search_kwargs = {"k": 4}

vector_search = DatabricksVectorSearch(
    index,
    text_column= text_column,
    columns = columns,
)
        
retriever_db = vector_search.as_retriever(search_kwargs=search_kwargs)
llm = ChatOpenAI(model="gpt-4o-mini")

#NODE FUNCTIONS


def question_decomposer(state: AgentState):
    original_question = state['original_question']
    prompt = prompts.DECOMPOSER_PROMPT
    prompt_template = ChatPromptTemplate.from_template(prompt)
    generate_queries = prompt_template | llm | StrOutputParser() | (lambda x: [q for q in x.split("\n") if q.strip()])
    response = generate_queries.invoke(original_question)
    return {"questions": response, "original_question": original_question}

def retriever(state: AgentState):
    
    questions = state['questions']
    documents = []

    for q in questions:
        documents.append(retriever_db.invoke(q))

    page_contents = []

    for docx_index, docx in enumerate(documents):
        page_contents.append([])
        for docy in docx:
            if not any(docy.page_content in sublist for sublist in page_contents):
                page_contents[docx_index].append(docy.page_content)

    return {"documents": page_contents}

async def grade_documents(state: AgentState):

    questions = state['questions']
    documents = state['documents']
    for q in questions:
        print(q)
        for d in documents:
            print(d)
    
    filtered_docs = []
    unfiltered_docs = []

    structured_retriever_grader = llm.with_structured_output(GradeDocuments)
    system = """You are a retriever grader checking if a document is relevant to a user's question.  
    If the document has words or meanings that helps to answer the question, mark it as relevant.  
    Give a simple 'yes' if the document is relevant or 'no' if the document is not relevant, remembering that the question and the document is all in portuguese."""
    grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n Retrieved document: \n\n {document}"),
    ]
    )
    retrieval_grader = grade_prompt | structured_retriever_grader
    for docx_index, (q, docx) in enumerate(zip(questions, documents)):
        filtered_docs.append([])
        unfiltered_docs.append([])
        for docy in docx:
            score = retrieval_grader.invoke({"question":q, "document":docy})
            grade = score.binary_score
            if grade=='yes':
                print(grade)
                print("---------DOCUMENTO RELEVANTE-----------")
                print(q)
                print(docy)
                filtered_docs[docx_index].append(docy)
            else:
                print(grade)
                print("---------DOCUMENTO NÃO RELEVANTE-----------")
                print(q)
                print(docy)
                unfiltered_docs[docx_index].append("FROM QUESTION: "+ q +"DOCUMENT: " + docy)
    empty_list = [i for i, sublista in enumerate(filtered_docs) if not sublista]
    refiltered_docs = [sublista for i, sublista in enumerate(filtered_docs) if i not in empty_list]
    refiltered_questions = [item for i, item in enumerate(questions) if i not in empty_list]

    return {"filter_documents": refiltered_docs, "unfilter_documents":unfiltered_docs, "questions": refiltered_questions}

def decide_to_generate(state:AgentState):
    
    filtered_documents = state['filter_documents']
    
    if filtered_documents:
        return "generate"
    else:
        return "transform_query"

def generate_answer(state: AgentState):

    prompt = prompts.GENERATE_ANSWER_PROMPT
    generate_answer_prompt = ChatPromptTemplate.from_template(prompt)
    answer_question = generate_answer_prompt | llm | StrOutputParser()

    questions=state['questions']
    documents=state['documents']

    reasonings = []
    for q, doc in zip(questions, documents):
        if len(doc) > 0:
            current_reasoning = answer_question.invoke({"context":doc,"question":q})
            reasonings.append(current_reasoning)

    return {"reasonings":reasonings}

def hallucination_router(state: AgentState):

  reasonings = state['reasonings']

  if reasonings:
    return "usefulco"
  else:
    return "not useful"

def grade_reasoning_vs_documents(state: AgentState):
    
    reasonings = state['reasonings']
    documents = state['documents']
    filtered_reasonings = []

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    system = prompts.REASONINGS_VS_DOCUMENTS_PROMPT
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n {generation}"),
        ]
    )
    hallucinations_grader = hallucination_prompt | structured_llm_grader
    for doc, reasoning in zip(documents, reasonings):
            score = hallucinations_grader.invoke({"documents":doc,"generation":reasoning})
            grade = score.binary_score
            if grade=='yes':
                filtered_reasonings.append(reasoning)
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")
    
    return {"reasonings": filtered_reasonings}

def transform_query(state: AgentState):

    question=state['original_question']
    documents=state['filter_documents']

    system = prompts.TRANSFORM_QUERY_PROMPT
     
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human","""Here is the initial question: \n\n {question} \n,
                Here is the document: \n\n {documents} \n ,
                Formulate an improved question. if possible other return 'question not relevant'."""
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    response = question_rewriter.invoke({"question":question,"documents":documents})
    if response == 'question not relevant':
        return {"documents":documents,"original_question":response,"generation":"question was not at all relevant"}
    else:   
        return {"documents":documents,"original_question":response}

def decide_to_generate_after_transformation(state: AgentState):
    question = state['original_question']
    
    if question=="question not relevant":
        return "query_not_at_all_relevant"
    else:
        return "Retriever"

def final_answer(state: AgentState):

    question = state['original_question']
    reasonings = state['reasonings']

    final_answer_template = prompts.FINAL_ANSWER_PROMPT
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)
    final_rag_chain = final_answer_prompt | llm | StrOutputParser()
    final_reasoning = final_rag_chain.invoke({"original_question": question, "qr_text": reasonings})

    return {"final_reasoning": final_reasoning}

def final_hallucination(state: AgentState):

  final_reasoning = state['final_reasoning']
  question = state['original_question']

  structured_answer_grader = llm.with_structured_output(GradeAnswer)
  system = prompts.FINAL_HALLUCINATION_PROMPT
  answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
  
  answer_grader = answer_prompt | structured_answer_grader
  score = answer_grader.invoke({"question":question,"generation":final_reasoning})
  grade = score.binary_score

  if grade=='yes':
    return "useful"
  else:
    return "not useful"

def supervisor(state: AgentState):
    question = state['original_question']
    return {"original_question": question}

def supervisor_choice(state: AgentState):

  question = state['original_question']

  structured_answer_grader = llm.with_structured_output(GradeAnswer)
  system = prompts.SUPERVISOR_CHOICE_PROMPT
  answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question}"),
        ]
    )
  answer_grader = answer_prompt | structured_answer_grader
  score = answer_grader.invoke({"question":question})
  grade = score.binary_score
  if grade=='yes':
    return "multiple"
  else:
    return "not multiple"

def final_multiple_choice_answer(state: AgentState):

    question = state['original_question']
    reasonings = state['reasonings']

    final_answer_template = prompts.FINAL_MULTIPLE_ANSWER_PROMPT
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)
    final_rag_chain = final_answer_prompt | llm | StrOutputParser()
    final_reasoning = final_rag_chain.invoke({"original_question": question, "qr_text": reasonings})

    return {"final_reasoning": final_reasoning}

# Define a new graph

builder = StateGraph(AgentState)

builder.add_node(question_decomposer)
builder.add_node(generate_answer)
builder.add_node(retriever)
builder.add_node(grade_documents)
builder.add_node(transform_query)
builder.add_node(grade_reasoning_vs_documents)
builder.add_node(final_answer)
builder.add_node(supervisor)
builder.add_node(final_multiple_choice_answer)

builder.add_edge("__start__", "question_decomposer")
builder.add_edge("question_decomposer", "retriever")
builder.add_edge("retriever", "grade_documents")
builder.add_conditional_edges("grade_documents",
                            decide_to_generate,
                            {
                            "generate": "generate_answer",
                            "transform_query": "transform_query"
                            }
                            )
builder.add_conditional_edges("transform_query",
                            decide_to_generate_after_transformation,
                            {
                            "Retriever":"question_decomposer",
                            "query_not_at_all_relevant": "__end__"
                            }
                            )
builder.add_edge("generate_answer", "grade_reasoning_vs_documents")
builder.add_conditional_edges("grade_reasoning_vs_documents",
                            hallucination_router,
                            {
                            "usefulco": "supervisor",
                            "not useful": "transform_query"
                            }
                            )
builder.add_conditional_edges("supervisor",
                            supervisor_choice,
                            {
                            "multiple": "final_multiple_choice_answer",
                            "not multiple": "final_answer"
                            }
                            )
builder.add_conditional_edges("final_answer",
                            final_hallucination,
                            {
                            "useful": "__end__",
                            "not useful": "transform_query"
                            }
                            )
builder.add_edge("final_multiple_choice_answer", "__end__")

graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
