from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. Medical assistant RAG generation prompt (used in rag_chain)
# ---------------------------------------------------------------------------

system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# 2. Router prompt — decides vectorstore vs web_search
# ---------------------------------------------------------------------------

router_system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at routing a user question to the right datasource.

The vectorstore contains medical knowledge: diseases, symptoms, treatments, medications,
anatomy, pharmacology, and general health information.

Route to 'vectorstore' for any medical/health question that could be answered from a
medical reference book or encyclopedia.

Route to 'web_search' for:
- Very recent medical news, drug approvals, or clinical trial results
- Non-medical questions
- Health topic that may not be in the medical reference (e.g. sexual health, lifestyle questions, mental health, emerging treatments)
- Questions about specific doctors, hospitals, or clinics

Return ONLY the routing decision — no explanation.""",
        ),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# 3. Retrieval grader prompt — is this doc relevant to the question?
# ---------------------------------------------------------------------------

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing the relevance of a retrieved medical document
to a user question.

If the document contains keywords or semantic meaning related to the question,
grade it as relevant. This does not need to be a strict test — the goal is to
filter out clearly irrelevant retrievals.
Return your answer ONLY as a JSON object with this format:
{{
  "binary_score": "yes" or "no"
}}
Do not explain anything.
""",
        ),
        (
            "human",
            "Retrieved document:\n\n{document}\n\nUser question: {question}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 4. Hallucination grader prompt — is the answer grounded in the facts?
# ---------------------------------------------------------------------------

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader checking whether an LLM-generated medical answer is
grounded in the retrieved source documents.

'yes' means the answer is supported by the documents.
'no' means the answer contains hallucinations or unsupported claims.

Be strict — patient safety depends on accuracy.""",
        ),
        (
            "human",
            "Source documents:\n\n{documents}\n\nLLM answer: {generation}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 5. Answer grader prompt — does the answer actually resolve the question?
# ---------------------------------------------------------------------------

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing whether a medical answer resolves the user's question.

'yes' means the answer directly addresses and resolves the question.
'no' means the answer is off-topic, incomplete, or does not address the question.""",
        ),
        (
            "human",
            "User question:\n\n{question}\n\nLLM answer: {generation}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 6. Query rewriter prompt — rewrite for better vectorstore retrieval
# ---------------------------------------------------------------------------

query_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query optimizer for a medical vectorstore retrieval system.

Given an input question, rewrite it into a cleaner, more precise version that will
retrieve better results from a medical knowledge base.

Focus on the core medical concept. Remove conversational filler.
Return ONLY the improved question — no explanation.""",
        ),
        (
            "human",
            "Original question:\n\n{question}\n\nImproved question:",
        ),
    ]
)
