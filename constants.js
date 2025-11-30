// Model constants
export const EMBEDDING_MODEL = "mxbai-embed-large";
export const AGENT_MODEL = "llama3.2:3b";
export const GENERATION_MODEL = "deepseek-r1:8b";

// Node names
export const AGENT_NODE = "agent";
export const RETRIEVE_NODE = "retrieve";
export const RELEVANCE_NODE = "assessRelevance";
export const QUERY_TRANSFORM_NODE = "transformQuery";
export const GENERATE_NODE = "generate";

// Tool names
export const RETRIEVER_TOOL_NAME = "retrieve_blog_posts";
export const GRADE_TOOL_NAME = "give_relevance_score";

// Decision constants
export const RELEVANT = "yes";
export const NOT_RELEVANT = "no";

// Prompts
const relevantValue = "yes";
const notRelevantValue = "no";

export const RELEVANCE_PROMPT_TEMPLATE = `You are a grader assessing relevance of retrieved docs to a user question.
Here are the retrieved docs:

-------

{context}

-------

Here is the user question: {question}

If the content of the docs are relevant to the users question, score them as relevant.
Give a binary score '${relevantValue}' or '${notRelevantValue}' score to indicate whether the docs are relevant to the question.
${relevantValue}: The docs are relevant to the question.
${notRelevantValue}: The docs are not relevant to the question.`;

export const REWRITE_PROMPT_TEMPLATE = `Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:

-------

{question}

-------

Formulate an improved question:`;

export const GENERATE_PROMPT_TEMPLATE = `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Here is the initial question:

-------

{question}

-------

Here is the context that you should use to answer the question:

-------

{context}

-------

Answer:`;