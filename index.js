import {OllamaEmbeddings} from "@langchain/ollama";
import {CheerioWebBaseLoader} from "@langchain/community/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {HNSWLib} from "@langchain/community/vectorstores/hnswlib";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import {Annotation} from "@langchain/langgraph";
import {BaseMessage, AIMessage, ToolMessage} from "@langchain/core/messages";
import {DynamicTool} from "@langchain/core/tools";
import {ToolNode} from "@langchain/langgraph/prebuilt";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {ChatOllama} from "@langchain/ollama";
import {END, StateGraph} from "@langchain/langgraph";
import {z} from "zod";
import {HumanMessage} from "@langchain/core/messages";
import {
    MODEL_EMBEDDING,
    MODEL_AGENT,
    MODEL_GENERATION,
    NODE_AGENT,
    NODE_RETRIEVE,
    NODE_RELEVANCE,
    NODE_TRANSFORM,
    NODE_GENERATE,
    TOOL_RETRIEVER,
    TOOL_GRADE,
    PROMPT_RELEVANCE,
    PROMPT_TRANSFORM,
    PROMPT_GENERATE,
    DECISION_RELEVANT,
    DECISION_NOT_RELEVANT
} from "./constants.js";
import {visualize} from "./workflow-visualizer.js";

const urls = [
    "https://deno.com/blog/not-using-npm-specifiers-doing-it-wrong",
    "https://deno.com/blog/v2.1",
    "https://deno.com/blog/build-database-app-drizzle",
];

async function main() {
    console.log("Starting RAG Agent with functions...");

    const embeddings = new OllamaEmbeddings({
        model: MODEL_EMBEDDING,
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

    const vectorStore = await HNSWLib.fromDocuments(
        allSplits,
        embeddings,
    );
    const retriever = vectorStore.asRetriever({
        k: 2, // a higher k value will provide more context, but will slow down the query
    });

    const tool = new DynamicTool({
        name: TOOL_RETRIEVER,
        description: "Search and return information about Deno from various blog posts.",
        func: async (query) => {
            const docs = await retriever.invoke(query);
            return docs.map(doc => doc.pageContent).join("\n\n");
        },
    });

    const tools = [tool];

    const callAgent = async (state) => {
        console.log("---CALL AGENT---");

        const {messages} = state;
        const filteredMessages = messages.filter((message) => {
            if (message instanceof AIMessage && message.tool_calls?.length) {
                return message.tool_calls[0].name !== TOOL_GRADE;
            }
            return true;
        });

        const model = new ChatOllama({
            model: MODEL_AGENT,
            temperature: 0,
            streaming: true,
        });

        const llmWithTools = model.bindTools(tools);
        const response = await llmWithTools.invoke(filteredMessages);
        return {
            messages: [response],
        };
    };

    const decideToRetrieveDocuments = (state) => {
        console.log("---DECIDE TO RETRIEVE---");
        const {messages} = state;
        const lastMessage = messages[messages.length - 1];

        if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
            console.log("---DECISION: RETRIEVE---");
            return NODE_RETRIEVE;
        }

        return END;
    };

    const gradeDocumentRelevance = async (state) => {
        console.log("---GRADE RELEVANCE---");

        const toolSchema = {
            name: TOOL_GRADE,
            description: "Grade document relevance with a binary score (yes/no).",
            schema: z.object({
                binaryScore: z.string().describe(`Relevance score '${DECISION_RELEVANT}' or '${DECISION_NOT_RELEVANT}'`),
            }),
        };

        const prompt = ChatPromptTemplate.fromTemplate(PROMPT_RELEVANCE);

        const llmWithTools = new ChatOllama({
            model: MODEL_AGENT,
            temperature: 0,
        }).bindTools([toolSchema]);

        const {messages} = state;
        const question = messages[0].content;
        const context = messages[messages.length - 1].content;

        const score = await prompt.pipe(llmWithTools).invoke({
            question,
            context,
        });

        return {
            messages: [score],
        };
    };

    const checkDocumentRelevance = (state) => {
        console.log("---CHECK RELEVANCE---");

        const {messages} = state;
        const lastMessage = messages[messages.length - 1];
        if (!(lastMessage instanceof AIMessage)) {
            throw new Error(
                "The 'checkDocumentRelevance' node requires the most recent message to be an AI message.",
            );
        }

        const {tool_calls: toolCalls} = lastMessage;
        if (!toolCalls || !toolCalls.length) {
            throw new Error(
                "The 'checkDocumentRelevance' node requires the most recent message to contain tool calls.",
            );
        }

        const relevant = toolCalls[0].args.binaryScore;
        console.log(`---DECISION: DOC RELEVANT=${relevant}`);
        return relevant;
    };

    const transformQuery = async (state) => {
        console.log("---TRANSFORM QUERY---");

        const {messages} = state;
        const question = messages[0].content;
        const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TRANSFORM);

        const model = new ChatOllama({
            model: MODEL_GENERATION,
            temperature: 0,
            streaming: true,
        });
        const response = await prompt.pipe(model).invoke({question});
        return {
            messages: [response],
        };
    };

    const generate = async (state) => {
        console.log("---GENERATE---");

        const {messages} = state;
        const question = messages[0].content;
        const lastToolMessage = messages.slice().reverse().find((msg) =>
            msg instanceof ToolMessage
        );
        if (!lastToolMessage) {
            throw new Error("No tool message found in the conversation history");
        }

        const context = lastToolMessage.content;

        const prompt = ChatPromptTemplate.fromTemplate(PROMPT_GENERATE);

        const llm = new ChatOllama({
            model: MODEL_GENERATION,
            temperature: 0,
            streaming: true,
        });

        const response = await prompt.pipe(llm).invoke({
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

    const toolNode = new ToolNode(tools);
    const workflow = new StateGraph(GraphState)
        .addNode(NODE_AGENT, callAgent)
        .addNode(NODE_RETRIEVE, toolNode)
        .addNode(NODE_RELEVANCE, gradeDocumentRelevance)
        .addNode(NODE_TRANSFORM, transformQuery)
        .addNode(NODE_GENERATE, generate);

    // Add edges and conditional edges
    workflow.addEdge("__start__", NODE_AGENT);

    // Decide whether to retrieve
    workflow.addConditionalEdges(
        NODE_AGENT,
        decideToRetrieveDocuments,
    );

    workflow.addEdge(NODE_RETRIEVE, NODE_RELEVANCE);

    // Edges taken after the `action' node is called.
    workflow.addConditionalEdges(
        NODE_RELEVANCE,
        checkDocumentRelevance,
        {
            [DECISION_RELEVANT]: NODE_GENERATE,
            [DECISION_NOT_RELEVANT]: NODE_TRANSFORM,
        },
    );

    workflow.addEdge(NODE_GENERATE, "__end__");
    workflow.addEdge(NODE_TRANSFORM, NODE_AGENT);


    const app = workflow.compile();
    visualize(app);

    console.log("RAG Agent is ready. Starting execution...");
    console.log("Question: What are some new features of Deno 2.1?");

    const inputs = {
        messages: [
            new HumanMessage("What are some new features of Deno 2.1?"),
        ],
    };

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
