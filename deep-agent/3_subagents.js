// Sub-agents implementation for context isolation
// JavaScript implementation based on 3_subagents.ipynb

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { createAgent } from "langchain";
import { BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";

// Define the state schema using Zod
const agentStateSchema = z.object({
  messages: z.array(z.any()).default([]),
  files: z.record(z.string(), z.string()).default({}),
});

// Task description prefix for the task tool
const TASK_DESCRIPTION_PREFIX = `Delegate a task to a specialized sub-agent with isolated context.

This creates a fresh context for the sub-agent containing only the task description,
preventing context pollution from the parent agent's conversation history.

Available sub-agents:
{other_agents}

Parameters:
- description: Clear, specific task or research question for the sub-agent
- subagent_type: Type of sub-agent to use (e.g., "research-agent")`;

// Define the SubAgent type
const SubAgentSchema = z.object({
  name: z.string(),
  description: z.string(),
  prompt: z.string(),
  tools: z.array(z.string()).optional(),
});

// Create the task delegation tool
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
      const isolatedState = {
        messages: [new HumanMessage(description)],
        files: {}
      };

      // Execute the sub-agent in isolation
      const result = await subAgent.invoke(isolatedState);

      // Return results to parent agent
      // Note: In this simplified implementation, we're returning the result directly
      // In a full implementation with proper state management, we'd use Command
      return result.messages[result.messages.length - 1]?.content || "No result returned";
    },
    {
      name: "task",
      description: TASK_DESCRIPTION_PREFIX.replace("{other_agents}", otherAgentsString),
      schema: z.object({
        description: z.string().describe("Clear, specific task or research question for the sub-agent"),
        subagent_type: z.string().describe("Type of sub-agent to use (e.g., 'research-agent')"),
      }),
    }
  );

  return taskTool;
}

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

// Add mock research instructions
const SIMPLE_RESEARCH_INSTRUCTIONS = `You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic.`;

// Create research sub-agent configuration
const researchSubAgent = {
  name: "research-agent",
  description: "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
  prompt: SIMPLE_RESEARCH_INSTRUCTIONS,
  tools: ["web_search"],
};

// Sub-agent usage instructions
const SUBAGENT_USAGE_INSTRUCTIONS = `You can delegate tasks to sub-agents.

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
tool calls in a single response to enable parallel execution. Use at most {max_concurrent_research_units} 
parallel agents per iteration.
</Available Tools>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards focused research** - Use single agent for simple questions, multiple only when clearly 
beneficial or when you have multiple independent research directions based on the user's request.
- **Stop when adequate** - Don't over-research; stop when you have sufficient information
- **Limit iterations** - Stop after {max_researcher_iterations} task delegations if you haven't found 
adequate sources
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
</Scaling Rules>`;

// Create supervisor agent with sub-agent capabilities
async function createSupervisorAgent() {
  const model = new ChatOllama({
    model: "qwen2.5:7b",
  });

  // Tools for sub-agents
  const subAgentTools = [web_search];

  // Create task tool to delegate tasks to sub-agents
  const taskTool = createTaskTool(
    subAgentTools, [researchSubAgent], model, agentStateSchema
  );

  // Tools for the supervisor
  const supervisorTools = [taskTool];

  // Create supervisor agent with system prompt
  const supervisorAgent = await createAgent({
    model,
    tools: supervisorTools,
    systemPrompt: SUBAGENT_USAGE_INSTRUCTIONS
      .replace("{max_concurrent_research_units}", "3")
      .replace("{max_researcher_iterations}", "3"),
    stateSchema: agentStateSchema,
  });

  return supervisorAgent;
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
  console.log("Creating supervisor agent with sub-agents capability...");
  
  try {
    const supervisorAgent = await createSupervisorAgent();
    console.log("Supervisor agent created successfully. Starting interaction...");

    // Example usage - the supervisor should delegate to sub-agents
    const result = await supervisorAgent.invoke({
      messages: [new HumanMessage("Give me an overview of Model Context Protocol (MCP).")],
    });

    console.log("Final state messages:");
    formatMessages(result.messages);
    
    console.log("\nInteraction completed.");
  } catch (error) {
    console.error("Error during agent execution:", error);
    console.error("Stack trace:", error.stack);
  }
}

main().catch(console.error);
