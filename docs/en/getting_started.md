# Project Setup

## 1. Cloning the Repository and Setting Up VSCode

First, clone the repository and open it in VSCode.

```bash
# Clone the repository using SSH key
git clone https://github.com/Tr0ubad0ur/multimodal-rag-gpt.git

# Open the project in VSCode
code multimodal-rag-gpt
```

Next, install the recommended VSCode extensions from the [.vscode/extensions.json] file (a popup will appear when opening the project).

After that, create a new branch or use an existing one according to the GitFlow workflow.

## 2. Creating a Virtual Environment

```bash
# Install the UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
. .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

> [!note] Glossary
> *pre-commit hooks* — scripts that run automatically before a commit to check or fix the code (linters, formatting, etc.), ensuring only clean code enters the repository.

```bash
# Install pre-commit hooks
uv run pre-commit install
```

```bash
# Install project dependencies
uv sync
```

## 3. Launch the Qdrant Vector Database

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 4. Project Structure

```bash
project-root/
│
├── backend/
│   ├── main.py
│   ├── api/
│   │   └── endpoints.py
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── image_embeddings.py
│   │   ├── llm.py
│   │   ├── vectordb.py
│   │   └── multimodal_rag.py
│   └── utils/
│       ├── loaders.py
│       └── config.py
│
├── data/
│   ├── ...
├── docs/
│   ├── en
│   └── ru
├── frontend
│   ├── ...
├── notebooks
│   └── workflow.ipynb
├── .env
├── .gitignore
├── .pre-commit-config
├── .python-version
├── mkdocks.yml
├── pyproject.toml
├── README.md
└── uv.lock

```
