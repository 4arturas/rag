import {OllamaEmbeddings} from "@langchain/ollama";
import {CheerioWebBaseLoader} from "@langchain/community/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {HNSWLib} from "@langchain/community/vectorstores/hnswlib";
import {ChatOllama} from "@langchain/ollama";
import {createAgent, dynamicSystemPromptMiddleware} from "langchain";
import {SystemMessage} from "@langchain/core/messages";
import {MODEL_AGENT, MODEL_EMBEDDING} from "../constants.js";

const url = "https://lilianweng.github.io/posts/2023-06-23-agent/";

async function main() {
    const docs = await new CheerioWebBaseLoader(
        url,
        {selector: "p"}
    ).load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const allSplits = await splitter.splitDocuments(docs);

    const embeddings = new OllamaEmbeddings({
        model: MODEL_EMBEDDING
    });

    const vectorStore = await HNSWLib.fromDocuments(allSplits, embeddings);

    const model = new ChatOllama({
        model: MODEL_AGENT
    });

    const agent = createAgent({
        model,
        tools: [],
        middleware: [
            dynamicSystemPromptMiddleware(async (state) => {
                // Get the last user query from the messages
                const lastQuery = state.messages[state.messages.length - 1].content;

                // Perform similarity search based on the query
                const retrievedDocs = await vectorStore.similaritySearch(lastQuery, 2);

                // Combine the content of retrieved documents
                const docsContent = retrievedDocs
                    .map((doc) => doc.pageContent)
                    .join("\n\n");

                return `You are a helpful assistant. Use the following context in your response:\n\n${docsContent}`;
            })
        ]
    });

    const inputMessage = `What is Task Decomposition?`;

    const chainInputs = {messages: [{role: "user", content: inputMessage}]};

    const stream = await agent.stream(chainInputs, {
        streamMode: "values",
    });

    for await (const step of stream) {
        const messages = step.messages;
        const lastMessage = messages[messages.length-1];
        const type = lastMessage.getType();
        const content = lastMessage.content;
        console.log(type, '->', content);
        console.log("-----\n");
    }

    return {agent, vectorStore};
}

main().catch(console.error);

