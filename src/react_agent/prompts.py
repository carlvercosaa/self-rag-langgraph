DECOMPOSER_PROMPT = """You are an expert in question decomposition.
    Your task is to break down the provided user question into an appropriate number of subquestions in Portuguese, based on its complexity, using vocabulary and phrasing relevant to the domain of criminal law.
    Do not generate more subquestions than necessary.
    The subquestions should reflect the key aspects or topics in the original question to help retrieve relevant documents from a vector database focused on criminal law.
    This process aims to overcome the limitations of distance-based similarity search.
    Provide the subquestions as a list, separated by newlines and without numbering.
    
    Original question: {question}
    """

document_grader_prompt = """You are a retriever grader checking if a document is relevant to a user's question.  
    If the document has words or meanings related to the question, mark it as relevant.  
    Give a simple 'yes' or 'no' answer to show if the document is relevant or not."""

GENERATE_ANSWER_PROMPT = """
    Question: {question}
    Context: {context}

    As a lawyer, in portuguese, let's think step by step about the issue exclusively through the context.

    I want the output in this format:
    Question: "Question provided""
    Reasoning: ""Generated reasoning""

    Avoid starting the response with introductory phrases like "De acordo com o contexto fornecido" or similar. Focus directly on the analysis and provide only the relevant information.
    """

REASONINGS_VS_DOCUMENTS_PROMPT = """You are a grader checking if an Reasoning is grounded in or supported by a set of retrieved facts.  
    Give a simple 'yes' or 'no' answer. 'Yes' means the reasoning is grounded in or supported by a set of retrieved the facts."""

TRANSFORM_QUERY_PROMPT = """You are a question re-writer that converts an input question into a better optimized version for vector store retrieval document.  
    You are given both a question and a document.  
    - First, check if the question is relevant to the document by identifying a connection or relevance between them.  
    - If there is a little relevancy, rewrite the question based on the semantic intent of the question and the context of the document.  
    - If no relevance is found, simply return this single word "question not relevant." dont return the entire phrase 
    Your goal is to ensure the rewritten question aligns well with the document for better retrieval."""

FINAL_ANSWER_PROMPT = """
    {qr_text}

    Main question: {original_question}

    As a lawyer, let's think step by step about the main issue exclusively through the reasonings provided, all in portuguese.
    """

FINAL_HALLUCINATION_PROMPT = """You are a grader assessing whether an reasoning addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

SUPERVISOR_CHOICE_PROMPT = """Classify the following question as multiple-choice or not.  
    A multiple-choice question presents at least two explicit answer choices, often in a list or separated by "or."
    Respond only with `"yes"` if it is multiple-choice or `"no"` otherwise. 
    """

FINAL_MULTIPLE_ANSWER_PROMPT = """
    {qr_text}
    Main question: {original_question}
    As a lawyer, we will think step by step about all the possible choices regarding the main issue exclusively through the reasons presented, all in Portuguese.
    """