import {StateGraph, START, END, Command, MemorySaver, interrupt, Annotation} from '@langchain/langgraph';
import {ChatOllama} from '@langchain/ollama';
import z from 'zod';

const llm = new ChatOllama({
    model: 'llama3.2:3b',
});

// 1. Define State Schema
const EmailStateAnnotation = Annotation.Root({
    emailContent: Annotation(),
    senderEmail: Annotation(),
    emailId: Annotation(),
    classification: Annotation(),
    ticketId: Annotation(),
    searchResults: Annotation(),
    customerHistory: Annotation(),
    draftResponse: Annotation(),
});

const EmailClassificationSchema = z.object({
    intent: z.enum(['question', 'bug', 'billing', 'feature', 'complex']),
    urgency: z.enum(['low', 'medium', 'high', 'critical']),
    topic: z.string(),
    summary: z.string(),
});

function readEmail(state) {
    console.log(`\n--- [READ EMAIL] ---`);
    console.log(`Processing: ${state.senderEmail}`);
    return {};
}

async function classifyIntent(state) {
    console.log('--- [CLASSIFYING] ---');
    const structuredLlm = llm.withStructuredOutput(EmailClassificationSchema);

    const classificationPrompt = `
    Analyze this customer email and classify it:
    Email: ${state.emailContent}
    From: ${state.senderEmail}
    Provide classification, including intent, urgency, topic, and summary.
    `;

    const classification = await structuredLlm.invoke(classificationPrompt);
    return {classification};
}

async function searchDocumentation(state) {
    console.log('--- [SEARCHING DOCS] ---');
    const classification = state.classification || {intent: 'question', topic: 'general'};
    return {
        searchResults: [
            `Doc for ${classification.intent}: Info about ${classification.topic}`,
            `Standard procedure for ${classification.topic}`
        ],
    };
}

async function bugTracking() {
    console.log('--- [CREATING TICKET] ---');
    return {ticketId: `BUG_${Math.floor(Math.random() * 10000)}`};
}

async function writeResponse(state) {
    console.log('--- [WRITING DRAFT] ---');
    const classification = state.classification;
    const context = (state.searchResults || []).map((d) => `- ${d}`).join('\n');

    const draftPrompt = `
    Draft a response to: ${state.emailContent}
    Intent: ${classification.intent}
    Docs:\n${context}
    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use documentation when relevant
    - Be brief    
    `;

    const response = await llm.invoke(draftPrompt);

    const needsReview = ['high', 'critical'].includes(classification.urgency) ||
        classification.intent === 'complex';
    const goto = needsReview ? 'human_review' : 'send_reply';

    console.log(needsReview ? 'Routing to Human Review...' : 'Routing to Send Reply...');

    return new Command({
        update: {draftResponse: response.content},
        goto,
    });
}

async function humanReview(state) {
    console.log('--- [INTERRUPTING FOR HUMAN REVIEW] ---');

    // The graph pauses here
    const decision = interrupt({
        action: 'Review Draft',
        draft: state.draftResponse,
    });

    if (decision.approved) {
        return new Command({
            update: {draftResponse: decision.editedResponse || state.draftResponse},
            goto: 'send_reply',
        });
    }
    return new Command({goto: END});
}

async function sendReply(state) {
    console.log('--- [SENDING REPLY] ---');
    // console.log(`Final Text: ${state.draftResponse.substring(0, 50)}...`);
    console.log(`Final Text: ${state.draftResponse}`);
    return {};
}

const memory = new MemorySaver();

export const graph = new StateGraph(EmailStateAnnotation)
    .addNode('read_email', readEmail)
    .addNode('classify_intent', classifyIntent)
    .addNode('search_documentation', searchDocumentation)
    .addNode('bug_tracking', bugTracking)
    .addNode('write_response', writeResponse, {
        ends: ['human_review', 'send_reply'],
    })
    .addNode('human_review', humanReview, {ends: ['send_reply', END]})
    .addNode('send_reply', sendReply)
    .addEdge(START, 'read_email')
    .addEdge('read_email', 'classify_intent')
    .addEdge('classify_intent', 'search_documentation')
    .addEdge('classify_intent', 'bug_tracking')
    .addEdge('search_documentation', 'write_response')
    .addEdge('bug_tracking', 'write_response')
    .addEdge('send_reply', END)
    .compile({checkpointer: memory}); // Memory is required for interrupts

async function main() {
    const config = {configurable: {thread_id: 'example-thread-1'}};

    const initialState = {
        emailContent: 'My car has blown up!',
        senderEmail: 'customer@test.com',
        emailId: '1',
    };

    console.log('Starting Workflow...');
    const result = await graph.invoke(initialState, config);

    // Check if the graph is waiting for a human
    const state = await graph.getState(config);
    if (state.next.length > 0 && state.next[0] === 'human_review') {
        console.log('\n[SYSTEM] Graph is paused. Simulating human approval...');

        // Resume the graph by providing the Command as input
        await graph.invoke(new Command({
            resume: {
                approved: true,
                editedResponse: 'Approved by Human: ' + state.values.draftResponse
            }
        }), config);

        console.log('\nWorkflow Complete.');
    } else {
        console.log('Workflow finished automatically.');
    }
}

main();
