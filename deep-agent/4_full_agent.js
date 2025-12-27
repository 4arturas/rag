// Implementation of the deep agent based on the 4_full_agent.ipynb notebook
// Using createAgent function as described in doc.md file
// Updated to use tavily package with proper API key and correct parameter format

import { createAgent } from "langchain";
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { BaseMessage, HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { TavilyClient } from 'tavily';

// Initialize Tavily client with the provided API key
const tavilyClient = new TavilyClient({apiKey:"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"});

// Define the state schema using Zod
const DeepAgentState = z.object({
  messages: z.array(z.any()).default([]),
  todos: z.array(z.object({
    content: z.string(),
    status: z.enum(["pending", "in_progress", "completed"]),
    id: z.string(),
  })).default([]),
  files: z.record(z.string(), z.string()).default({}),
});

// Tool descriptions
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

const READ_TODOS_DESCRIPTION = `
Read the current TODO list from the agent state.

This tool retrieves the current list of tasks with their statuses to help
the agent maintain awareness of its progress and plan next steps.

## When to Use
- Before starting a new phase of work to review existing tasks
- After completing a task to confirm status updates
- When planning next steps to understand current progress
- To refresh task context after interruptions

## Returns
Current list of TODO items with their content and status.
`;

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
          id: z.string(),
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
    description: READ_TODOS_DESCRIPTION,
    schema: z.object({}),
  }
);

// File system tools
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

// Virtual file system state
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

// Tavily search tool - search the web and save results to files
const tavily_search = tool(
  async ({ query, max_results = 1, topic = "general" }) => {
    try {
      // Execute search using the correct API format based on the client implementation
      const response = await tavilyClient.search({
        query: query,
        max_results: max_results,
        topic: topic,
        include_raw_content: true
      });
      
      // Process the results
      const results = response.results || [];
      const processedResults = [];
      
      for (const result of results) {
        // Create a unique filename based on the result
        const url = new URL(result.url);
        const domain = url.hostname.replace('www.', '');
        const timestamp = Date.now().toString().substr(-4); // Last 4 digits of timestamp
        const filename = `${domain.replace(/\./g, '_')}_${timestamp}.md`;
        
        // Create file content
        const fileContent = `# Search Result: ${result.title}

**URL:** ${result.url}
**Query:** ${query}
**Date:** ${new Date().toISOString().split('T')[0]}

## Content
${result.content}

## Raw Content
${result.raw_content || 'No raw content available'}

---
*Retrieved using Tavily search API*
`;
        
        // Save to virtual file system
        virtualFileSystem[filename] = fileContent;
        processedResults.push({
          title: result.title,
          url: result.url,
          summary: result.content.substring(0, 200) + (result.content.length > 200 ? "..." : ""),
          filename: filename
        });
      }
      
      // Create summary for the tool response
      const summaries = processedResults.map(r => `- ${r.filename}: ${r.summary}`).join('\n');
      return `Found ${processedResults.length} result(s) for '${query}':\n\n${summaries}\n\nFiles saved: ${processedResults.map(r => r.filename).join(', ')}`;
    } catch (error) {
      console.error("Tavily search error:", error);
      return `Error performing search: ${error.message}`;
    }
  },
  {
    name: "tavily_search",
    description: "Search the web for information and save detailed results to files while returning minimal context. Performs web search and saves full content to files for context offloading. Returns only essential information to help the agent decide on next steps.",
    schema: z.object({
      query: z.string().describe("Search query to execute. Be specific and clear about what information you're looking for."),
      max_results: z.number().optional().default(1).describe("Maximum number of results to return (default: 1)"),
      topic: z.enum(["general", "news", "finance"]).optional().default("general").describe("Topic filter - 'general', 'news', or 'finance' (default: 'general')"),
    }),
  }
);

// Think tool for reflection
const think_tool = tool(
  async ({ reflection }) => {
    return `Reflection recorded: ${reflection}`;
  },
  {
    name: "think_tool",
    description: "Tool for strategic reflection on research progress and decision-making.",
    schema: z.object({
      reflection: z.string().describe("Your detailed reflection on research progress, findings, gaps, and next steps")
    })
  }
);

// Task tool for sub-agent delegation
function createTaskTool(tools, subagents, model, stateSchema) {
  // Create agent registry
  const agents = {};

  // Build tool name mapping for selective tool assignment
  const toolsByName = {};
  tools.forEach(tool_ => {
    toolsByName[tool_.name] = tool_;
  });

  // Create specialized sub-agents based on configurations
  subagents.forEach(agent => {
    let agentTools;
    if (agent.tools) {
      // Use specific tools if specified
      agentTools = agent.tools.map(toolName => toolsByName[toolName]);
    } else {
      // Default to all tools
      agentTools = tools;
    }

    agents[agent.name] = createAgent({
      model,
      tools: agentTools,
      systemPrompt: agent.prompt,
      stateSchema: stateSchema,
    });
  });

  // Generate description of available sub-agents for the tool description
  const otherAgentsString = subagents.map(agent =>
    `- ${agent.name}: ${agent.description}`
  ).join("\n");

  // Create the task tool
  const taskTool = tool(
    async ({ description, subagent_type }) => {
      // Validate requested agent type exists
      if (!agents[subagent_type]) {
        return `Error: invoked agent of type ${subagent_type}, the only allowed types are [${Object.keys(agents).map(k => `\`${k}\``).join(", ")}]`;
      }

      // Get the requested sub-agent
      const subAgent = agents[subagent_type];

      // Create isolated context with only the task description
      // This is the key to context isolation - no parent history
      const state = {
        messages: [new HumanMessage(description)],
        files: {}
      };

      // Execute the sub-agent in isolation
      const result = await subAgent.invoke(state);

      // Return results to parent agent
      return result.messages[result.messages.length - 1]?.content || "No result returned";
    },
    {
      name: "task",
      description: `Delegate a task to a specialized sub-agent with isolated context.

This creates a fresh context for the sub-agent containing only the task description,
preventing context pollution from the parent agent's conversation history.

Available sub-agents:
${otherAgentsString}

Parameters:
- description: Clear, specific task or research question for the sub-agent
- subagent_type: Type of sub-agent to use (e.g., "research-agent")`,
      schema: z.object({
        description: z.string().describe("Clear, specific task or research question for the sub-agent"),
        subagent_type: z.string().describe("Type of sub-agent to use (e.g., 'research-agent')"),
      }),
    }
  );

  return taskTool;
}

// Create research sub-agent
const researchSubAgent = {
  name: "research-agent",
  description: "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
  prompt: `You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question.
You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 1-2 search tool calls maximum
- **Normal queries**: Use 2-3 search tool calls maximum
- **Very Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>`,
  tools: ["tavily_search", "think_tool"],
};

// Create task tool to delegate tasks to sub-agents
const taskTool = createTaskTool(
  [tavily_search, think_tool], [researchSubAgent], new ChatOllama({ model: "qwen2.5:7b" }), DeepAgentState
);

// Create the agent
async function createDeepAgent() {
  const model = new ChatOllama({
    model: "qwen2.5:7b",
  });

  const tools = [ls, read_file, write_file, write_todos, read_todos, tavily_search, think_tool, taskTool];

  // Create agent with system prompt
  const agent = await createAgent({
    model,
    tools,
    systemPrompt: `You are a research assistant that uses tools to gather information and answer questions.
    
# TODO MANAGEMENT
Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan.
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Always create a research plan of TODOs and conduct research following the above guidelines for ANY user request.
IMPORTANT: Aim to batch research tasks into a *single TODO* in order to minimize the number of TODOs you have to keep track of.

# FILE SYSTEM USAGE
You have access to a virtual file system to help you retain and save context.

## Workflow Process
1. **Orient**: Use ls() to see existing files before starting work
2. **Save**: Use write_file() to store the user's request so that we can keep it for later
3. **Research**: Proceed with research. The search tool will write files.
4. **Read**: Once you are satisfied with the collected sources, read the files and use them to answer the user's question directly.

# SUB-AGENT DELEGATION
You can delegate tasks to sub-agents.

<Task>
Your role is to coordinate research by delegating specific research tasks to sub-agents.
</Task>

<Available Tools>
1. **task(description, subagent_type)**: Delegate research tasks to specialized sub-agents
   - description: Clear, specific research question or task
   - subagent_type: Type of agent to use (e.g., "research-agent")
2. **think_tool(reflection)**: Reflect on the results of each delegated task and plan next steps.
   - reflection: Your detailed reflection on the results of the task and next steps.

**PARALLEL RESEARCH**: When you identify multiple independent research directions, make multiple **task**
tool calls in a single response to enable parallel execution. Use at most 3 parallel agents per iteration.
</Available Tools>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards focused research** - Use single agent for simple questions, multiple only when clearly
beneficial or when you have multiple independent research directions based on the user's request.
- **Stop when adequate** - Don't over-research; stop when you have sufficient information
- **Limit iterations** - Stop after 3 task delegations if you haven't found adequate sources
</Hard Limits>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: "List the top 10 coffee shops in San Francisco" → Use 1 sub-agent, store in
\`findings_coffee_shops.md\`

**Comparisons** can use a sub-agent for each element of the comparison:
- *Example*: "Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety" → Use 3 sub-agents
- Store findings in separate files: \`findings_openai_safety.md\`, \`findings_anthropic_safety.md\`,
\`findings_deepmind_safety.md\`

**Multi-faceted research** can use parallel agents for different aspects:
- *Example*: "Research renewable energy: costs, environmental impact, and adoption rates" → Use 3 sub-agents
- Organize findings by aspect in separate files

**Important Reminders:**
- Each **task** call creates a dedicated research agent with isolated context
- Sub-agents can't see each other's work - provide complete standalone instructions
- Use clear, specific language - avoid acronyms or abbreviations in task descriptions
</Scaling Rules>`,
    stateSchema: DeepAgentState,
  });

  return agent;
}

// Export the agent creation function
export { createDeepAgent, DeepAgentState, virtualFileSystem };

// Example usage
async function runExample() {
  console.log("Creating deep agent...");
  const agent = await createDeepAgent();
  console.log("Agent created successfully!");
  
  // Example invocation
  const result = await agent.invoke({
    messages: [new HumanMessage("Give me an overview of Model Context Protocol (MCP).")],
  });
  
  console.log("Result:", result);
}

runExample();