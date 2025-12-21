import {OllamaEmbeddings} from "@langchain/ollama";
import {CheerioWebBaseLoader} from "@langchain/community/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {HNSWLib} from "@langchain/community/vectorstores/hnswlib";
import {ChatOllama} from "@langchain/ollama";
import {z} from "zod";
import {tool} from "@langchain/core/tools";
import {createAgent} from "langchain";
import {HumanMessage, SystemMessage} from "@langchain/core/messages";
import {MODEL_AGENT, MODEL_EMBEDDING} from "./constants.js";

const url = "https://lilianweng.github.io/posts/2023-06-23-agent/";

async function main() {
    const loader = new CheerioWebBaseLoader(
        url,
        {selector: "p"}
    );

    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const allSplits = await splitter.splitDocuments(docs);
    const embeddings = new OllamaEmbeddings({
        model: MODEL_EMBEDDING
    });
    const model = new ChatOllama({
        model: MODEL_AGENT,
    });

    const vectorStore = await HNSWLib.fromDocuments(allSplits, embeddings);


    const retrieve = tool(
        async ({query}) => {
            const retrievedDocs = await vectorStore.similaritySearch(query, 2);
            const serialized = retrievedDocs
                .map(
                    (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent.substring(0, 500)}...`
                )
                .join("\n");

            console.log(`🔍 Retrieved ${retrievedDocs.length} documents for query: "${query}"`);
            return [serialized, retrievedDocs];
        },
        {
            name: "retrieve",
            description: "Retrieve information related to a query.",
            schema: z.object({query: z.string()}),
            responseFormat: "content_and_artifact",
        }
    );


    const tools = [retrieve];
    const systemPrompt = new SystemMessage(
        `You have access to a tool that retrieves context from a blog post.
        Use the tool to help answer user queries.`
    );

    const agent = createAgent({
        model,
        tools,
        systemPrompt
    });

    const inputMessage = `
    What is the standard method for Task Decomposition?
    Once you get the answer, look up common extensions of that method.
    `;

    const agentInputs = {
        messages: [
            new HumanMessage(inputMessage),
        ],
    };

    const stream = await agent.stream(
        agentInputs,
        {
            streamMode: "values"
        }
    );

    for await (const step of stream) {
        const messages = step.messages;
        const lastMessage = messages[messages.length-1];
        const type = lastMessage.getType();
        const content = lastMessage.content;
        console.log(type, '->', content);
        console.log("-----\n");
    }

    return {agent, vectorStore, retrieve};
}

main().catch(console.error);