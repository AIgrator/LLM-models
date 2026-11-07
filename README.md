# LLM Models List Generator

This repository contains an automated workflow to generate a curated `models.json` file for the Gemini AI Assistant desktop application.

## How It Works

1.  **Scheduled Execution**: A GitHub Actions workflow runs automatically every 6 hours.
2.  **Data Fetching**: A Python script (`scripts/update_models.py`) fetches the latest model data from the OpenRouter API and the official XAI API.
3.  **Filtering & Curation**: The script applies a series of advanced filters to select only high-quality, multimodal models that are suitable for the assistant's needs. It excludes preview versions, specialized models, and those with insufficient context length.
4.  **Structuring**: The final list is structured into a clean `models.json` file, organizing models by provider.
5.  **Deployment**: The generated `models.json` is automatically published to GitHub Pages, providing a stable, public URL for the main application to consume.

This approach ensures that the AI assistant always has access to an up-to-date list of the best available models without requiring an application update.
