
import os
import json
from pathlib import Path

def generate_puter_html(visual_plan, output_dir):
    """
    Generates an HTML file that uses Puter.js to generate images client-side.
    This avoids API keys and backend limits by leveraging the User-Pays (free tier) model.
    """
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nano Banana Visuals Dashboard</title>
    <script src="https://js.puter.com/v2/"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
        .prompt { font-size: 0.9rem; color: #94a3b8; margin-bottom: 1rem; font-style: italic; }
        .image-container { min-height: 300px; display: flex; align-items: center; justify-content: center; background: #0f172a; border-radius: 8px; overflow: hidden;}
        img { width: 100%; height: auto; display: block; }
        h2 { color: #facc15; font-size: 1.25rem; margin-top: 0; }
        button { background: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; margin-top: 1rem;}
        button:hover { background: #2563eb; }
        .status { margin-top: 0.5rem; font-size: 0.8rem; color: #64748b; }
    </style>
</head>
<body>
    <h1>🍌 Nano Banana Visuals (Puter.js)</h1>
    <p>Generating high-fidelity infographics using <b>Gemini 2.5 Flash Image Preview</b> & <b>Flux.1 Schnell</b>.</p>
    
    <div class="grid" id="grid">
        <!-- Cards injected here -->
    </div>

    <script>
        const visualPlan = VISUAL_PLAN_PLACEHOLDER;

        const container = document.getElementById('grid');

        // Render Cards
        visualPlan.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'card';
            
            const prompt = item.image_generation_prompt || item.Prompt || item.description;
            const title = item['Concept Name'] || item.ConceptName || `Visual #${index + 1}`;

            card.innerHTML = `
                <h2>${title}</h2>
                <div class="prompt">"${prompt}"</div>
                <div class="image-container" id="img-container-${index}">
                    <span class="status">Waiting to generate...</span>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 10px;">
                    <button onclick="generateImage(${index}, '${prompt.replace(/'/g, "\\'")}', 'gemini-2.5-flash-image-preview')">Generate (Nano Banana)</button>
                    <button onclick="generateImage(${index}, '${prompt.replace(/'/g, "\\'")}', 'black-forest-labs/FLUX.1-schnell')">Generate (Flux)</button>
                </div>
            `;
            container.appendChild(card);
            
            // Auto-trigger Gemini generation
            setTimeout(() => generateImage(index, prompt, 'gemini-2.5-flash-image-preview'), index * 1000); // Staggered start
        });

        function generateImage(index, prompt, model) {
            const imgContainer = document.getElementById(`img-container-${index}`);
            imgContainer.innerHTML = '<span class="status">Generating with ' + model + '...</span>';
            
            puter.ai.txt2img(prompt, { model: model })
                .then(img => {
                    imgContainer.innerHTML = '';
                    imgContainer.appendChild(img);
                })
                .catch(err => {
                    imgContainer.innerHTML = '<span class="status" style="color: #ef4444">Error: ' + err.message + '</span>';
                });
        }
    </script>
</body>
</html>
    """
    
    # Inject JSON
    safe_json = json.dumps(visual_plan)
    final_html = html_content.replace("VISUAL_PLAN_PLACEHOLDER", safe_json)
    
    filename = "visuals_dashboard.html"
    filepath = Path(output_dir) / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    return filepath
