import {ChatOllama} from '@langchain/ollama';
import z from 'zod';

const llm = new ChatOllama({
    model: 'llama3.2:3b',
});

let EmailState = {
    emailContent: null,
    senderEmail: null,
    emailId: null,
    classification: null,
    ticketId: null,
    searchResults: [],
    customerHistory: null,
    draftResponse: null,
};

const EmailClassificationSchema = z.object({
    intent: z.enum(['question', 'bug', 'billing', 'feature', 'complex']),
    urgency: z.enum(['low', 'medium', 'high', 'critical']),
    topic: z.string(),
    summary: z.string(),
});

async function requestHumanReview(draft) {
    console.log('--- [WAITING FOR HUMAN REVIEW] ---');
    console.log(`Reviewing Draft: "${draft.substring(0, 50)}..."`);

    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                approved: true,
                editedResponse: `Approved by Human: ${draft}`
            });
        }, 1000);
    });
}

async function runEmailWorkflow(state) {
    console.log(`\n--- [STARTING WORKFLOW: ${state.emailId}] ---`);

    console.log(`
    --- [READ EMAIL] --- 
    Processing: ${state.senderEmail}
    ${state.emailContent}
    `);

    console.log('--- [CLASSIFYING] ---');
    const structuredLlm = llm.withStructuredOutput(EmailClassificationSchema);

    const classificationPrompt = `
    Analyze this customer email and classify it:
    Email: ${state.emailContent}
    From: ${state.senderEmail}
    Provide classification, including intent, urgency, topic, and summary.
    `;

    const classification = await structuredLlm.invoke(classificationPrompt);
    state.classification = classification;
    console.log(`Detected Intent: ${classification.intent} | Urgency: ${classification.urgency}`);

    if (state.classification.intent === 'bug') {
        console.log('--- [ROUTING: BUG TRACKING] ---');
        state.ticketId = `BUG_${Math.floor(Math.random() * 10000)}`;
    } else {
        console.log('--- [ROUTING: DOC SEARCH] ---');
        state.searchResults = [
            `Doc for ${state.classification.intent}: Info about ${state.classification.topic}`,
            `Standard procedure for ${state.classification.topic}`
        ];
    }

    console.log('--- [WRITING DRAFT] ---');
    const context = (state.searchResults || []).map((d) => `- ${d}`).join('\n');

    const draftPrompt = `
    Draft a response to: ${state.emailContent}
    Intent: ${state.classification.intent}
    Docs:\n${context}
    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use documentation when relevant
    - Be brief    
    `;

    const response = await llm.invoke(draftPrompt);
    state.draftResponse = response.content;

    const needsReview = ['high', 'critical'].includes(state.classification.urgency) ||
        state.classification.intent === 'complex';

    if (needsReview) {
        console.log('--- [ROUTING: HUMAN REVIEW REQUIRED] ---');

        const decision = await requestHumanReview(state.draftResponse);

        if (!decision.approved) {
            console.log('--- [REVIEW REJECTED] ---');
            state.draftResponse = "Rejected by human review";
            return state;
        }

        state.draftResponse = decision.editedResponse || state.draftResponse;
    }

    await sendReply(state);
    return state;
}

async function sendReply(state) {
    console.log('--- [SENDING REPLY] ---');
    console.log(`Final Text: ${state.draftResponse}`);
    return {};
}

async function main() {
    const testCases = [
        {
            id: 'TEST_QUESTION',
            email: 'How do I export my data to a CSV format?',
            sender: 'user1@example.com'
        },
        {
            id: 'TEST_BUG',
            email: 'The login button is completely unresponsive.',
            sender: 'user2@example.com'
        },
        {
            id: 'TEST_BILLING',
            email: 'I was charged twice for my subscription this month. Please refund.',
            sender: 'user3@example.com'
        },
        {
            id: 'TEST_FEATURE',
            email: 'It would be great if you added a dark mode to the dashboard.',
            sender: 'user4@example.com'
        },
        {
            id: 'TEST_COMPLEX',
            email: 'I need a custom enterprise integration that supports our proprietary SSO and legacy database from 1998.',
            sender: 'user5@example.com'
        }
    ];

    const test = testCases[1];
    const initialState = {
        ...EmailState,
        emailContent: test.email,
        senderEmail: test.sender,
        emailId: test.id,
        searchResults: []
    };

    try {
        await runEmailWorkflow(initialState);
        console.log(`-------------------------------------------`);
    } catch (error) {
        console.error(`Error in test ${test.id}:`, error);
    }

}

main();