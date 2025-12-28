import {z} from "zod";
import {ChatOllama} from "@langchain/ollama";

// Tool Constants
const SEARCH_WEB = "search_web";
const TELL_JOKE = "tell_joke";
const GET_WEATHER = "get_weather";
const CONVERT_CURRENCY = "convert_currency";
const DEBUG_CODE = "debug_code";
const SUMMARIZE_TEXT = "summarize_text";

const ToolSchema = z.object({
    tool: z.enum([
        SEARCH_WEB,
        TELL_JOKE,
        GET_WEATHER,
        CONVERT_CURRENCY,
        DEBUG_CODE,
        SUMMARIZE_TEXT
    ]),
    reasoning: z.string().describe("Brief explanation of why this tool was chosen")
})

async function main() {
    const llm = new ChatOllama({
        model: "qwen2.5:7b",
        temperature: 0,
    });

    const structuredLlm = llm.withStructuredOutput(ToolSchema);

    const prompts = [
        "Tell me a joke about a programmer and a lightbulb",
        "What are the latest open-source LLM benchmarks for 2025?",
        "What is the current temperature in Tokyo?",
        "How many Euros can I get for 500 US Dollars?",
        "Why is my Python script throwing a KeyError when I access this dictionary?",
        "Can you give me a short version of this long article about space exploration?",
        "Forecast for New York this weekend",
        "Explain the difference between Llama 3 and Qwen 2.5"
    ];

    console.log(`--- Starting Tool Classification for ${prompts.length} prompts ---\n`);

    for (const prompt of prompts) {
        const classification = await structuredLlm.invoke(prompt);
        console.log(`Prompt: "${prompt}"`);
        console.log(`Action: [${classification.tool.toUpperCase()}]`);
        console.log(`Reason: ${classification.reasoning}`);
        console.log("=".repeat(50));
    }
}

main();