import { OllamaEmbeddings } from "@langchain/ollama";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { DynamicTool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOllama } from "@langchain/ollama";
import { isAIMessage, isToolMessage } from "@langchain/core/messages";
import { END, StateGraph } from "@langchain/langgraph";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import {
  EMBEDDING_MODEL,
  AGENT_MODEL,
  GENERATION_MODEL,
  AGENT_NODE,
  RETRIEVE_NODE,
  GRADE_DOCS_NODE,
  REWRITE_NODE,
  GENERATE_NODE,
  RETRIEVER_TOOL_NAME,
  GRADE_TOOL_NAME,
  GRADE_PROMPT_TEMPLATE,
  REWRITE_PROMPT_TEMPLATE,
  GENERATE_PROMPT_TEMPLATE
} from "./constants.js";

const urls = [
    "https://deno.com/blog/not-using-npm-specifiers-doing-it-wrong",
    "https://deno.com/blog/v2.1",
    "https://deno.com/blog/build-database-app-drizzle",
];

async function main() {
  console.log("Starting RAG Agent with functions...");

  const embeddings = new OllamaEmbeddings({
    model: EMBEDDING_MODEL,
  });


  const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load()),
  );
  const docsList = docs.flat();
  console.log(`Loaded ${docsList.length} documents`);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const allSplits = await splitter.splitDocuments(docsList);
  console.log(`Split blog posts into ${allSplits.length} sub-documents.`);

  // Create vector store and retriever
  const vectorStore = await HNSWLib.fromDocuments(
    allSplits,
    embeddings,
  );
  // const retriever = vectorStore.asRetriever();
  const retriever = vectorStore.asRetriever({
      k: 2, // a higher k value will provide more context, but will slow down the query
  });

  const tool = new DynamicTool({
    name: RETRIEVER_TOOL_NAME,
    description: "Search and return information about Deno from various blog posts.",
    func: async (query) => {
      const docs = await retriever.invoke(query);
      return docs.map(doc => doc.pageContent).join("\n\n");
    },
  });

  const tools = [tool];

  const toolNode = new ToolNode(tools);

  const agent = async (state) => {
    console.log("---CALL AGENT---");

    const { messages } = state;
    const filteredMessages = messages.filter((message) => {
      if (isAIMessage(message) && message.tool_calls?.length) {
        return message.tool_calls[0].name !== GRADE_TOOL_NAME;
      }
      return true;
    });

    const model = new ChatOllama({
      model: AGENT_MODEL,
      temperature: 0,
      streaming: true,
    });

    const llmWithTools = model.bindTools(tools);
    const response = await llmWithTools.invoke(filteredMessages);
    return {
      messages: [response],
    };
  };

  const shouldRetrieve = (state) => {
    console.log("---DECIDE TO RETRIEVE---");
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];

    if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
      console.log("---DECISION: RETRIEVE---");
      return RETRIEVE_NODE;
    }

    return END;
  };

  const gradeDocuments = async (state) => {
    console.log("---GET RELEVANCE---");

    const toolSchema = {
      name: GRADE_TOOL_NAME,
      description: "Give a relevance score to the retrieved documents.",
      schema: z.object({
        binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
      }),
    };

    const prompt = ChatPromptTemplate.fromTemplate(GRADE_PROMPT_TEMPLATE);

    const model = new ChatOllama({
      model: AGENT_MODEL,
      temperature: 0,
    }).bindTools([toolSchema]);

    const { messages } = state;
    const firstMessage = messages[0];
    const lastMessage = messages[messages.length - 1];

    const chain = prompt.pipe(model);

    const score = await chain.invoke({
      question: firstMessage.content,
      context: lastMessage.content,
    });

    return {
      messages: [score],
    };
  };

  const checkRelevance = (state) => {
    console.log("---CHECK RELEVANCE---");

    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    if (!isAIMessage(lastMessage)) {
      throw new Error(
        "The 'checkRelevance' node requires the most recent message to be an AI message.",
      );
    }

    const { tool_calls: toolCalls } = lastMessage;
    if (!toolCalls || !toolCalls.length) {
      throw new Error(
        "The 'checkRelevance' node requires the most recent message to contain tool calls.",
      );
    }

    if (toolCalls[0].args.binaryScore === "yes") {
      console.log("---DECISION: DOCS RELEVANT---");
      return "yes";
    }
    console.log("---DECISION: DOCS NOT RELEVANT---");
    return "no";
  };

  const rewrite = async (state) => {
    console.log("---TRANSFORM QUERY---");

    const { messages } = state;
    const question = messages[0].content;
    const prompt = ChatPromptTemplate.fromTemplate(REWRITE_PROMPT_TEMPLATE);

    const model = new ChatOllama({
      model: GENERATION_MODEL,
      temperature: 0,
      streaming: true,
    });
    const response = await prompt.pipe(model).invoke({ question });
    return {
      messages: [response],
    };
  };

  const generate = async (state) => {
    console.log("---GENERATE---");

    const { messages } = state;
    const question = messages[0].content;
    // Extract the most recent ToolMessage
    const lastToolMessage = messages.slice().reverse().find((msg) =>
      isToolMessage(msg)
    );
    if (!lastToolMessage) {
      throw new Error("No tool message found in the conversation history");
    }

    const context = lastToolMessage.content;

    const prompt = ChatPromptTemplate.fromTemplate(GENERATE_PROMPT_TEMPLATE);

    const llm = new ChatOllama({
      model: GENERATION_MODEL,
      temperature: 0,
      streaming: true,
    });

    const ragChain = prompt.pipe(llm);

    const response = await ragChain.invoke({
      context,
      question,
    });

    return {
      messages: [response],
    };
  };

    const GraphState = Annotation.Root({
        messages: Annotation({
            reducer: (x, y) => x.concat(y),
            default: () => [],
        }),
    });

  const workflow = new StateGraph(GraphState)
    .addNode(AGENT_NODE, agent)
    .addNode(RETRIEVE_NODE, toolNode)
    .addNode(GRADE_DOCS_NODE, gradeDocuments)
    .addNode(REWRITE_NODE, rewrite)
    .addNode(GENERATE_NODE, generate);

  // Add edges and conditional edges
  workflow.addEdge("__start__", AGENT_NODE);

  // Decide whether to retrieve
  workflow.addConditionalEdges(
    AGENT_NODE,
    shouldRetrieve,
  );

  workflow.addEdge(RETRIEVE_NODE, GRADE_DOCS_NODE);

  // Edges taken after the `action' node is called.
  workflow.addConditionalEdges(
    GRADE_DOCS_NODE,
    checkRelevance,
    {
      yes: GENERATE_NODE,
      no: REWRITE_NODE,
    },
  );

  workflow.addEdge(GENERATE_NODE, "__end__");
  workflow.addEdge(REWRITE_NODE, AGENT_NODE);

  const app = workflow.compile();
  console.log("RAG Agent is ready. Starting execution...");
  console.log("Question: What are some new features of Deno 2.1?");

  // Prepare input
  const inputs = {
    messages: [
      new HumanMessage("What are some new features of Deno 2.1?"),
    ],
  };

  // Run the workflow
  let finalState;
  for await (const output of await app.stream(inputs)) {
    for (const [key, value] of Object.entries(output)) {
      console.log(`${key} -->`);
      finalState = value;
    }
  }

  const lastMessage = finalState.messages[finalState.messages.length - 1];
  const content = lastMessage.content;

  console.log("\n\nGenerated output from agent:");
  console.log(content);

  return app;
}

main().catch(console.error);
