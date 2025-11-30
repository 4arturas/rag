import { OllamaEmbeddings } from "@langchain/ollama";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
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
    GENERATE_PROMPT_TEMPLATE, RELEVANT, NOT_RELEVANT
} from "./constants.js";

const urls = [
    "https://deno.com/blog/not-using-npm-specifiers-doing-it-wrong",
    "https://deno.com/blog/v2.1",
    "https://deno.com/blog/build-database-app-drizzle",
];

async function main()
{
    const embeddings = new OllamaEmbeddings({model:EMBEDDING_MODEL});
    const docs = await Promise.all(
        urls.map( url => new CheerioWebBaseLoader(url).load() )
    );
    const docsList = docs.flat();
    const splitter = new RecursiveCharacterTextSplitter({chunkSize:500,chunkOverlap:50});
    const splits = await splitter.splitDocuments( splitter );
    const vectorStore = await MemoryVectorStore.fromDocuments( splits, embeddings );
    const retriever = vectorStore.asRetriever();
    const tool = new DynamicTool({
        name: RETRIEVER_TOOL_NAME,
        description: "Search and return information about Deno from various blog posts.",
        func: async ( query ) =>
        {
            const docs = await retriever.invoke( query );
            return docs.map( doc => doc.pageContent ).join("\n");
        }
    });
    const tools = [tool];

   const agent = async ( state ) =>
   {
       console.log("---CALL AGENT---");
       const { messages } = state;
       const filteredMessages = messages.filter( msg => {
           if ( isAIMessage(msg) && msg.tool_calls?.length )
               return msg.tool_calls[0].name !== GRADE_TOOL_NAME;
           return true;
       });
       const llmWithTools = new ChatOllama({
           model: AGENT_MODEL,
           temperature: 0,
           streaming: true
       }).bindTools(tools);
       const response = await llmWithTools.invoke( filteredMessages );
       return {
           messages: [response]
       }
   }

   const shouldRetrieve = async ( state ) =>
   {
       console.log("---DECIDE TO RETRIEVE---");
       const { messages } = state;
       const lastMessage = messages[messages.length-1];
       if ( isAIMessage(lastMessage) && lastMessage.tool_calls?.length )
       {
           console.log("---DECISION: RETRIEVE---");
           return RETRIEVE_NODE;
       }
       return END;
   }

   const gradeDocuments = async ( state ) =>
   {
       console.log("---GET RELEVANCE---");
       const toolSchema = {
           name: GRADE_TOOL_NAME,
           description: "Give a relevance score to the retrieved documents.",
           schema: z.object({
               binaryScore: z.string().describe(`Relevance score '${RELEVANT}' or '${NOT_RELEVANT}'`)
           })
       };
       const llmWithTools = new ChatOllama({
           model: AGENT_MODEL,
           temperature: 0
       }).bindTools([toolSchema]);
       const { messages } = state;
       const question = messages[0];
       const context = messages[messages.length-1];
       const prompt = ChatPromptTemplate.fromTemplate(GRADE_PROMPT_TEMPLATE);
       const score = await prompt.pipe(llmWithTools).invoke({
           question,
           context
       });
       return {
           messages: [score]
       }
   }
}