import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { z } from 'zod';
import { ChatOllama } from '@langchain/ollama';
import { MODEL_QWEN } from '../constants.js';

// Define the schema for a single task based on your tasks.json structure.
const TaskSchema = z.object({
    id: z.number(),
    title: z.string(),
    description: z.string(),
    details: z.string(),
    testStrategy: z.string(),
    priority: z.enum(["high", "medium", "low"]),
    dependencies: z.array(z.number()),
    status: z.literal("pending").or(z.literal("completed")),
    subtasks: z.array(z.number()).default([]),
    updatedAt: z.string(),
    result: z.string().optional().default("")
});

const TaskListSchema = z.object({
    tasks: z.array(TaskSchema)
});

const model = MODEL_QWEN;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Serve tasks.json file
app.get('/tasks.json', (req, res) => {
    const tasksFilePath = path.join(__dirname, 'public', 'tasks.json');
    res.sendFile(tasksFilePath);
});

app.post('/generate-tasks', async (req, res) => {
    const { prd } = req.body;

    if (!prd) {
        return res.status(400).json({ error: 'PRD content is required' });
    }

    try {
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
        6. Focus only on actual code generation tasks, not project setup or final testing tasks.
        7. Avoid creating tasks for "project setup", "initial configuration", "environment setup", or "final testing/deployment" - instead create tasks for the actual features and functionality.
        `;

        const userPrompt = `
        Convert the following PRD into a JSON array of development tasks:

        PRD CONTENT:
        ${prd}
        `;

        console.log(`--- Analyzing PRD and Generating Tasks using ${model} ---`);

        const response = await structuredLlm.invoke([
            ["system", systemPrompt],
            ["human", userPrompt]
        ]);

        // Save the generated tasks to tasks.json file
        try {
            const fs = await import('fs');
            const tasksFilePath = path.join(__dirname, 'public', 'tasks.json');
            await fs.promises.writeFile(tasksFilePath, JSON.stringify(response.tasks, null, 2));
            console.log(`Tasks saved to ${tasksFilePath}`);
        } catch (saveError) {
            console.error("Failed to save tasks to file:", saveError.message);
        }

        res.json(response);
    } catch (error) {
        console.error("Failed to generate tasks with Ollama:", error.message);
        res.status(500).json({ error: 'Failed to generate tasks: ' + error.message });
    }
});

// Route to implement a task (update its status to completed)
app.post('/implement-task', async (req, res) => {
    const { taskId } = req.body;

    if (!taskId) {
        return res.status(400).json({ success: false, error: 'Task ID is required' });
    }

    try {
        const fs = await import('fs');
        const tasksFilePath = path.join(__dirname, 'public', 'tasks.json');

        let tasks = [];

        try {
            const tasksData = await fs.promises.readFile(tasksFilePath, 'utf8');
            tasks = JSON.parse(tasksData);
        } catch (readError) {
            if (readError.code === 'ENOENT') {
                return res.status(404).json({ success: false, error: 'Tasks file not found.' });
            } else {
                throw readError;
            }
        }

        const taskToComplete = tasks.find(task => task.id === taskId);
        if (!taskToComplete) {
            return res.status(404).json({ success: false, error: 'Task not found' });
        }

        try {
            const previousImplementations = tasks
                .filter(t => t.status === 'completed' && t.result)
                .map(t => `### Task ${t.id}: ${t.title}\n${t.result}`)
                .join("\n\n");

            // TODO: read tasks.json, task with id 1 and 2 have wrong result attribute, llm is adding ```javascript...```, it should return only code
            const implementationSystemPrompt = `
            You are a Senior Software Engineer. Your task is to implement the requested feature based on the provided details.

            CONTEXT OF PREVIOUS WORK:
            Below is the code already implemented for previous tasks. You MUST ensure your new code is compatible with this existing codebase.

            ${previousImplementations || "No tasks have been implemented yet."}

            CRITICAL RULES:
            1. Return ONLY the code or HTML requested.
            2. Do NOT include any conversational text, explanations, or "Here is your code" messages.
            3. Ensure the code is production-ready, well-commented, and follows best practices.
            4. If the task is UI-related, return valid HTML/JSX.
            5. If the task is logic-related, return valid JavaScript.
            6. Do not wrap the output in markdown code blocks (\`\`\`) unless specifically asked for a file format that requires it.
            7. Focus only on generating actual application code, not project setup, configuration, or testing tasks.
            8. If the task is about project setup, dependencies, or testing, generate the actual code that would be needed for those features.
            9. Boilerplate code:
            ${boilerplateCode}
            `;

            const implementationUserPrompt = `
            NOW IMPLEMENTING:
            TASK TITLE: ${taskToComplete.title}
            DESCRIPTION: ${taskToComplete.description}
            IMPLEMENTATION DETAILS: ${taskToComplete.details}
            TEST STRATEGY: ${taskToComplete.testStrategy}

            Implement this task now. Output only the implementation code.
            `;

            const llm = new ChatOllama({
                model: "qwen2.5-coder:7b",
                temperature: 0
            });

            const result = await llm.invoke([
                ["system", implementationSystemPrompt],
                ["human", implementationUserPrompt]
            ]);

            // Strip markdown code block markers from the result
            let cleanedResult = result.content || "No code generated";

            // Remove markdown code block markers (```javascript, ```js, ```, etc.)
            // cleanedResult = cleanedResult.replace(/^```[a-zA-Z]*\n?/g, '');
            // cleanedResult = cleanedResult.replace(/\n?```$/g, '');

            taskToComplete.result = cleanedResult;
        } catch (modelError) {
            console.error("Failed to call model for task result:", modelError.message);
            taskToComplete.result = "// Model processing failed. Please try again.";
        }

        // Update the task status to completed
        taskToComplete.status = 'completed';
        taskToComplete.updatedAt = new Date().toISOString();

        // Save updated tasks to file
        await fs.promises.writeFile(tasksFilePath, JSON.stringify(tasks, null, 2));

        res.json({
            success: true,
            updatedTasks: tasks,
            taskResult: taskToComplete.result
        });
    } catch (error) {
        console.error("Failed to update task status:", error.message);
        res.status(500).json({ success: false, error: 'Failed to update task status: ' + error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

const boilerplateCode = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>React + Ant Design – full-screen tabs</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/antd/5.26.5/reset.min.css"/>
    <style>body{padding:30px}</style>

    <script src="https://cdn.jsdelivr.net/npm/dayjs@1/dayjs.min.js"><\/script>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"><\/script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"><\/script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"><\/script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/antd/5.26.7/antd.min.js"><\/script>
</head>
<body>
!!! ADD CODE HERE !!!
<\/script>
</body>
</html>
    `;