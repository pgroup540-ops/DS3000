import pandas as pd
data = [
    {
        "Model Name": "Llama 3.1 8B",
        "Developer": "Meta",
        "Parameter Count": "8 Billion",
        "Context Window": "128k tokens",
        "Key Features": "Multilingual (8+ languages), Tool Use, Long Context, 15T training tokens",
        "Description": "A dense transformer model designed for efficiency and long-context reasoning. It excels in multilingual dialogue and agentic workflows, bridging the gap between edge-device latency and server-grade performance."
    },
    {
        "Model Name": "Gemma 2 9B",
        "Developer": "Google",
        "Parameter Count": "9 Billion",
        "Context Window": "8k tokens",
        "Key Features": "Knowledge Distillation, Sliding Window Attention, Soft-capping, High performance-to-size ratio",
        "Description": "Built on the same research and technology as Gemini. It utilizes knowledge distillation from larger models to achieve outsized performance for its weight class, particularly in reasoning and coding tasks."
    },
    {
        "Model Name": "Mistral 7B (v0.3)",
        "Developer": "Mistral AI",
        "Parameter Count": "7.3 Billion",
        "Context Window": "32k tokens",
        "Key Features": "Sliding Window Attention (SWA), Grouped-Query Attention (GQA), Rolling Buffer Cache",
        "Description": "A highly efficient model known for its speed and low memory footprint. Its architectural innovations (SWA/GQA) allow it to handle longer sequences with lower compute costs, making it a favorite for local deployment."
    },
    {
        "Model Name": "Qwen 2.5 7B",
        "Developer": "Alibaba Cloud",
        "Parameter Count": "7.61 Billion",
        "Context Window": "128k tokens",
        "Key Features": "Multilingual (29+ languages), Structured Output (JSON), Coding & Math specialization",
        "Description": "A powerhouse for coding and mathematics that supports over 29 languages. It is specifically optimized to follow complex instructions and generate structured data formats like JSON, making it ideal for enterprise applications."
    },
    {
        "Model Name": "Phi-3 Mini",
        "Developer": "Microsoft",
        "Parameter Count": "3.8 Billion",
        "Context Window": "128k tokens",
        "Key Features": "High-quality Synthetic Data Training, Mobile Deployment, Reasoning-dense",
        "Description": "A lightweight model trained on a 'textbook-quality' dataset of 3.3 trillion tokens. Despite its small size (deployable on phones), it rivals larger models in logic and reasoning benchmarks."
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = "llm_models_comparison.xlsx"
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Successfully created {output_file}")