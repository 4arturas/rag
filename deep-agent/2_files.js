// Virtual file system tools for agent state management
// JavaScript implementation based on 2_files.ipynb
// Using createAgent as per LangGraph v1 migration guide

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { createAgent } from "langchain";
import { BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";

// Tool descriptions
const LS_DESCRIPTION = `List all files in the virtual filesystem stored in agent state.

Shows what files currently exist in agent memory. Use this to orient yourself before other file operations
and maintain awareness of your file organization.

No parameters required - simply call ls() to see all available files.`;

const READ_FILE_DESCRIPTION = `Read content from a file in the virtual filesystem with optional pagination.

This tool returns file content with line numbers (like \`cat -n\`) and supports reading large files in chunks
to avoid context overflow.

Parameters:
- file_path (required): Path to the file you want to read
- offset (optional, default=0): Line number to start reading from
- limit (optional, default=2000): Maximum number of lines to read

Essential before making any edits to understand existing content. Always read a file before editing it.`;

const WRITE_FILE_DESCRIPTION = `Create a new file or completely overwrite an existing file in the virtual filesystem.

This tool creates new files or replaces entire file contents. Use for initial file creation or complete
rewrites. Files are stored persistently in agent state.

Parameters:
- file_path (required): Path where the file should be created/overwritten
- content (required): The complete content to write to the file

Important: This replaces the entire file content.`;

// Global variable to hold the virtual file system state
let virtualFileSystem = {};

// ls tool - list all files in the virtual filesystem
const ls = tool(
  async () => {
    const fileNames = Object.keys(virtualFileSystem);
    if (fileNames.length === 0) {
      return "No files in the virtual filesystem";
    }
    return fileNames.join(", ");
  },
  {
    name: "ls",
    description: LS_DESCRIPTION,
    schema: z.object({}),
  }
);

// read_file tool - read content from a file in the virtual filesystem
const read_file = tool(
  async ({ file_path, offset = 0, limit = 2000 }) => {
    if (!virtualFileSystem[file_path]) {
      return `Error: File '${file_path}' not found`;
    }

    const content = virtualFileSystem[file_path];
    if (!content) {
      return "System reminder: File exists but has empty contents";
    }

    const lines = content.split("\n");
    const startIdx = offset;
    const endIdx = Math.min(startIdx + limit, lines.length);

    if (startIdx >= lines.length) {
      return `Error: Line offset ${offset} exceeds file length (${lines.length} lines)`;
    }

    const resultLines = [];
    for (let i = startIdx; i < endIdx; i++) {
      const lineContent = lines[i].substring(0, 2000); // Truncate long lines
      resultLines.push(`${(i + 1).toString().padStart(6)}\t${lineContent}`);
    }

    return resultLines.join("\n");
  },
  {
    name: "read_file",
    description: READ_FILE_DESCRIPTION,
    schema: z.object({
      file_path: z.string().describe("Path to the file to read"),
      offset: z.number().optional().default(0).describe("Line number to start reading from"),
      limit: z.number().optional().default(2000).describe("Maximum number of lines to read"),
    }),
  }
);

// write_file tool - write content to a file in the virtual filesystem
const write_file = tool(
  async ({ file_path, content }) => {
    virtualFileSystem[file_path] = content;
    return `Updated file ${file_path} with ${content.length} characters`;
  },
  {
    name: "write_file",
    description: WRITE_FILE_DESCRIPTION,
    schema: z.object({
      file_path: z.string().describe("Path where the file should be created/overwritten"),
      content: z.string().describe("Content to write to the file"),
    }),
  }
);

// File usage instructions
const FILE_USAGE_INSTRUCTIONS = `You have access to a virtual file system to help you retain and save context.

## Workflow Process
1. **Orient**: Use ls() to see existing files before starting work
2. **Save**: Use write_file() to store the user's request so that we can keep it for later
3. **Read**: Once you are satisfied with the collected sources, read the saved file and use it to ensure that you directly answer the user's question.`;

// Add mock research instructions
const SIMPLE_RESEARCH_INSTRUCTIONS = `IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question.`;

// Full prompt
const INSTRUCTIONS = `${FILE_USAGE_INSTRUCTIONS}\n\n${"=".repeat(80)}\n\n${SIMPLE_RESEARCH_INSTRUCTIONS}`;

// Mock search result
const search_result = `The Model Context Protocol (MCP) is an open standard protocol developed 
by Anthropic to enable seamless integration between AI models and external systems like 
tools, databases, and other services. It acts as a standardized communication layer, 
allowing AI models to access and utilize data from various sources in a consistent and 
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants 
to external services by providing a unified language for data exchange.`;

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

// Define the agent state schema using Zod
const agentStateSchema = z.object({
  messages: z.array(z.any()).default([]),
});

// Create agent with file system tools
async function createFileAgent() {
  const model = new ChatOllama({
    model: "qwen2.5:7b",
  });
  
  const tools = [ls, read_file, write_file, web_search];

  // Create agent with system prompt
  const agent = await createAgent({
    model,
    tools,
    systemPrompt: INSTRUCTIONS,
    stateSchema: agentStateSchema,
  });

  return agent;
}

// Format and display messages
function formatMessages(messages) {
  if (!messages || !Array.isArray(messages)) {
    console.log("No messages to display");
    return;
  }
  
  messages.forEach((msg) => {
    if (msg._getType && msg._getType() === "human") {
      console.log(`\n👤 Human: ${msg.content}`);
    } else if (msg._getType && msg._getType() === "ai") {
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
    } else if (msg._getType && msg._getType() === "tool") {
      const toolMsg = msg;
      console.log(`\n🔧 Tool Output: ${msg.content}`);
      if (toolMsg.tool_call_id) {
        console.log(`   Tool Call ID: ${toolMsg.tool_call_id}`);
      }
    } else {
      // For other message types
      if (msg.content) {
        console.log(`\n📄 Message: ${msg.content}`);
      }
    }
  });
}

// Example usage
async function main() {
  console.log("Creating file system agent with createAgent...");
  
  try {
    const agent = await createFileAgent();
    console.log("Agent created successfully. Starting interaction...");

    // Example usage - the agent should now use the file system tools
    const result = await agent.invoke({
      messages: [new HumanMessage("Give me an overview of Model Context Protocol (MCP). First, check what files exist using ls(), then save my request to a file called 'user_request.txt', search for information about MCP, and finally read back the saved file to answer my question.")],
    });

    console.log("Result object keys:", Object.keys(result || {}));
    console.log("Final state messages:");
    formatMessages(result?.messages);
    
    console.log("\nVirtual file system contents:", virtualFileSystem);
    console.log("\nInteraction completed.");
  } catch (error) {
    console.error("Error during agent execution:", error);
    console.error("Stack trace:", error.stack);
  }
}

main().catch(console.error);