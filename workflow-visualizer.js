import { writeFileSync, existsSync, mkdirSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function generateMermaid(workflow, title = "LangGraph Workflow", options = {}) {
  try {
    const graph = workflow.getGraph();
    const mermaidSyntax = graph.drawMermaid();
    return mermaidSyntax;
  } catch (error) {
    console.error("Error generating Mermaid diagram:", error);
    throw error;
  }
}

function saveMermaidToFile(mermaidSyntax, filename = "workflow", directory = ".") {
  if (!existsSync(directory)) {
    mkdirSync(directory, { recursive: true });
  }

  const mmdFilePath = path.join(directory, `${filename}.mmd`);
  writeFileSync(mmdFilePath, mermaidSyntax);
}

function generateHtmlVisualization(mermaidSyntax, title = "LangGraph Workflow Visualization", filename = "workflow", directory = ".") {
  if (!existsSync(directory)) {
    mkdirSync(directory, { recursive: true });
  }

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
    <title>${title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .diagram-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 20px 0;
            overflow-x: auto;
        }
        .mermaid {
            text-align: center;
            min-height: 400px;
        }
        .info-panel {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }
        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 20px 0;
        }
        .code-block pre {
            margin: 0;
            font-size: 14px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 14px;
        }
        .controls {
            text-align: center;
            margin: 15px 0;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
        }
        button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>${title}</h1>
        <div class="subtitle">Generated LangGraph Workflow Visualization</div>

        <div class="info-panel">
            <strong>How to use this visualization:</strong>
            <ul>
                <li>Refresh the page to regenerate the diagram if needed</li>
                <li>You can copy the Mermaid syntax below and paste it into the <a href="https://mermaid.live" target="_blank">Mermaid Live Editor</a> for additional customization</li>
                <li>Right-click on the diagram to save as an image</li>
                <li>Zoom in/out using Ctrl + Scroll or pinch gestures</li>
            </ul>
        </div>

        <div class="diagram-container">
            <div class="mermaid">
                ${mermaidSyntax}
            </div>
        </div>

        <div class="controls">
            <button onclick="location.reload()">Refresh Diagram</button>
            <button onclick="toggleTheme()">Toggle Theme</button>
        </div>

        <h2>Mermaid Syntax</h2>
        <div class="code-block">
            <pre><code>${mermaidSyntax.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
        </div>

        <div class="footer">
            Generated with LangGraph Workflow Visualizer | ${new Date().toLocaleDateString()}
        </div>
    </div>

    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            },
            fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
            fontSize: 14
        });

        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.setAttribute('data-theme', newTheme);

            // Update mermaid theme
            const newMermaidTheme = newTheme === 'dark' ? 'dark' : 'default';
            mermaid.initialize({
                startOnLoad: true,
                theme: newMermaidTheme,
                securityLevel: 'loose',
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                },
                fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                fontSize: 14
            });

            // Re-render the diagram
            location.reload();
        }
    </script>
</body>
</html>`;

  const htmlFilePath = path.join(directory, `${filename}.html`);
  writeFileSync(htmlFilePath, htmlContent);
}

export async function visualizeWorkflow(workflow, title = "LangGraph Workflow", options = {}) {
  const {
    filename = "workflow",
    directory = ".",
    saveMmd = true,
    saveHtml = true,
    showInConsole = true
  } = options;

  try {

    const mermaidSyntax = await generateMermaid(workflow, title);

    if (saveMmd) {
      await saveMermaidToFile(mermaidSyntax, filename, directory);
    }

    if (saveHtml) {
      await generateHtmlVisualization(mermaidSyntax, title, filename, directory);
    }

    return {
      mermaidSyntax,
      filesCreated: {
        mmd: saveMmd ? path.join(directory, filename + '.mmd') : null,
        html: saveHtml ? path.join(directory, filename + '.html') : null
      }
    };
  } catch (error) {
    console.error("Error in visualization workflow:", error);
    throw error;
  }
}

export async function visualize(workflow, title = "LangGraph Workflow", filename = "workflow") {
  return await visualizeWorkflow(workflow, title, {
    filename,
    directory: ".",
    saveMmd: true,
    saveHtml: true,
    showInConsole: true
  });
}