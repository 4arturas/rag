import { BaseMessage, HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { createAgent } from "langchain";

// Tool description for write_todos
const WRITE_TODOS_DESCRIPTION = `
Create and manage structured task lists for tracking progress through complex workflows.

## When to Use
- Multi-step or non-trivial tasks requiring coordination
- When user provides multiple tasks or explicitly requests todo list
- Avoid for single, trivial actions unless directed otherwise

## Structure
- Maintain one list containing multiple todo objects (content, status, id)
- Use clear, actionable content descriptions
- Status must be: pending, in_progress, or completed

## Best Practices
- Only one in_progress task at a time
- Mark completed immediately when task is fully done
- Always send the full updated list when making changes
- Prune irrelevant items to keep list focused

## Progress Updates
- Call TodoWrite again to change task status or edit content
- Reflect real-time progress; don't batch completions
- If blocked, keep in_progress and add new task describing blocker

## Parameters
- todos: List of TODO items with content and status fields

## Returns
Updates agent state with new todo list.
`;

// TODO usage instructions
const TODO_USAGE_INSTRUCTIONS = `
Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan.
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Always create a research plan of TODOs and conduct research following the above guidelines for ANY user request.
IMPORTANT: Aim to batch research tasks into a *single TODO* in order to minimize the number of TODOs you have to keep track of.
`;

// Mock search result
const search_result = `The Model Context Protocol (MCP) is an open standard protocol developed
by Anthropic to enable seamless integration between AI models and external systems like
tools, databases, and other services. It acts as a standardized communication layer,
allowing AI models to access and utilize data from various sources in a consistent and
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants
to external services by providing a unified language for data exchange.`;

// write_todos tool
const write_todos = tool(
  async ({ todos }) => {
    return `Updated todo list to ${JSON.stringify(todos)}`;
  },
  {
    name: "write_todos",
    description: WRITE_TODOS_DESCRIPTION,
    schema: z.object({
      todos: z.array(
        z.object({
          content: z.string(),
          status: z.enum(["pending", "in_progress", "completed"]),
        })
      ),
    }),
  }
);

// read_todos tool
const read_todos = tool(
  async () => {
    // This will be handled by the agent state, returning placeholder
    return "No todos currently in the list.";
  },
  {
    name: "read_todos",
    description: "Read the current TODO list from the agent state.",
    schema: z.object({}),
  }
);


// Mock search tool
const web_search = tool(
  async ({ query }) => {
    return search_result;
  },
  {
    name: "web_search",
    description: "Search the web for information on a specific topic.",
    schema: z.object({
      query: z.string(),
    }),
  }
);

// Format and display messages
function formatMessages(messages) {
  messages.forEach((msg) => {
    if (msg._getType() === "human") {
      console.log(`\n👤 Human: ${msg.content}`);
    } else if (msg._getType() === "ai") {
      const aiMsg = msg;
      if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
        aiMsg.tool_calls.forEach((toolCall) => {
          console.log(`\n🔧 Tool Call: ${toolCall.name}`);
          console.log(`   Args: ${JSON.stringify(toolCall.args, null, 2)}`);
          console.log(`   ID: ${toolCall.id}`);
        });
      } else {
        console.log(`\n🤖 AI: ${msg.content}`);
      }
    } else if (msg._getType() === "tool") {
      const toolMsg = msg;
      console.log(`\n🔧 Tool Name: ${msg.name}`);
      console.log(`   Tool Output: ${msg.content}`);
      console.log(`   Tool Call ID: ${toolMsg.tool_call_id}`);
    }
  });
}

const customAgentState = z.object({
  todos: z.array(z.object({
    content: z.string(),
    status: z.enum(["pending", "in_progress", "completed"]),
    id: z.string(),
  })).default([]),
});

async function main() {
  try {
    // Create agent using createAgent (LangGraph v1 approach)
    const model = new ChatOllama({
      model: "qwen2.5:7b",
    });
    const tools = [write_todos, web_search, read_todos];

    // Add mock research instructions
    const SIMPLE_RESEARCH_INSTRUCTIONS = "IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question.";

    // Create agent with system prompt
    const agent = await createAgent({
      model,
      tools,
      systemPrompt: TODO_USAGE_INSTRUCTIONS + "\n\n" + "=".repeat(80) + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS,
      stateSchema: customAgentState,
    });

    console.log("Agent created successfully. Starting interaction...");

    // Example usage
    const result = await agent.invoke({
      messages: [new HumanMessage("Give me a short summary of the Model Context Protocol (MCP).")],
    });

    formatMessages(result.messages);

    // Log final result
    console.log("\nInteraction completed.");
  } catch (error) {
    console.error("Error in main function:", error);
  }
}

main();
