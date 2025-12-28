import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { MODEL_QWEN } from "../constants.js";

/**
 * Define the schema for a single task based on your tasks.json structure.
 */
const TaskSchema = z.object({
    id: z.number(),
    title: z.string(),
    description: z.string(),
    details: z.string(),
    testStrategy: z.string(),
    priority: z.enum(["high", "medium", "low"]),
    dependencies: z.array(z.number()),
    status: z.literal("pending"),
    subtasks: z.array(z.number()).default([]),
    updatedAt: z.string()
});

const TaskListSchema = z.object({
    tasks: z.array(TaskSchema)
});

const model = MODEL_QWEN;

/**
 * Imitates the implementation of tasks one by one.
 * Ensures that a task only "starts" after all its dependencies are met.
 */
async function imitateTaskImplementation(tasks) {
    console.log("\n--- Starting Task Implementation Simulation ---");
    const completedTasks = new Set();
    const taskQueue = [...tasks];

    while (taskQueue.length > 0) {
        // Find tasks whose dependencies are all met
        const readyTaskIndex = taskQueue.findIndex(task =>
            task.dependencies.every(depId => completedTasks.has(depId))
        );

        if (readyTaskIndex === -1) {
            console.error("Simulation Error: Circular dependency or missing dependency detected.");
            break;
        }

        const task = taskQueue.splice(readyTaskIndex, 1)[0];

        console.log(`[IN PROGRESS] Task ${task.id}: ${task.title}`);

        // Simulate "work" being done
        await new Promise(resolve => setTimeout(resolve, 800));

        console.log(`[COMPLETED]   Task ${task.id}: ${task.title}`);
        console.log(`              Details: ${task.details.substring(0, 60)}...`);
        console.log(`              Testing: ${task.testStrategy}`);
        console.log('--------------------------------------------');

        completedTasks.add(task.id);
    }

    console.log("--- All tasks have been successfully implemented! ---\n");
}

async function generateTasksFromPRD(prdText) {
    const llm = new ChatOllama({
        model: model,
        temperature: 0,
    });

    const structuredLlm = llm.withStructuredOutput(TaskListSchema);
    const currentTimestamp = new Date().toISOString();

    const systemPrompt = `
    You are an expert Project Manager and Technical Architect. 
    Your goal is to take a PRD (Product Requirements Document) and break it down into a logical sequence of development tasks.
    
    Rules:
    1. Tasks must be sequential and atomic.
    2. Include dependencies (numeric IDs).
    3. Each task must include a detailed 'testStrategy' and implementation 'details'.
    4. All tasks must have the status 'pending'.
    5. Set 'updatedAt' for every task to exactly: ${currentTimestamp}
    `;

    const userPrompt = `
    Convert the following PRD into a JSON array of development tasks:
    
    PRD CONTENT:
    ${prdText}
    `;

    console.log(`--- Analyzing PRD and Generating Tasks using ${model} ---`);

    try {
        const response = await structuredLlm.invoke([
            ["system", systemPrompt],
            ["human", userPrompt]
        ]);

        console.log("Successfully generated tasks:");
        console.log(JSON.stringify(response, null, 2));

        // Implementation of the TODO: Imitate task processing
        if (response && response.tasks) {
            await imitateTaskImplementation(response.tasks);
        }

        return response;

    } catch (error) {
        console.error("Failed to generate tasks with Ollama:", error.message);
    }
}

const prdContent = `
Project: Basic Task Tracker
Core Features:
1. Add tasks with a text description.
2. View a list of all active tasks.
3. Delete tasks from the list.
4. Persist data using in memory json.
Tech Stack: React, Ant Design, Single HTML File.
`;

generateTasksFromPRD(prdContent);