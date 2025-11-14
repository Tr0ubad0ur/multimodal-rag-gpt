<div align="center">
    <a href="https://www.mirea.ru">
      <img src="https://www.mirea.ru/upload/medialibrary/205/yly02h0ioocdeega8ir1kbsstul6q9ws/new_logo.png" width="96" alt="РТУ МИРЭА">
    </a>
    <h1>Diploma</h1>
    <p><i>ИПТИП, Fullstack-разработка, ЭФБО-04-22</i></p>
    <p>
        <a href="https://t.me/Papajunn" target="_blank">Матвей Вишняков</a>
    </p>
</div>

# Diploma

## 1 Клонирование репозитория и настройка VSCode

Для начала требуется склонировать репозиторий и открыть его.

```bash
# Клонирование репозитория с использованием SSH ключа
git clone https://github.com/Tr0ubad0ur/multimodal-rag-gpt.git

# Открытие проекта в VSCode
code multimodal-rag-gpt
```

Теперь требуется установить рекомендуемые расширения VSCode из файла [`.vscode/extensions.json`] (при открытии проекта появится всплывающее окно).

Далее требуется сделать новую ветку, либо использовать существующую согласно GitFlow процессу работы.

## 2 Создание виртуального окружения

```bash
# Установка менеджера пакетов UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" на Windows
```

```bash
# Создание виртуальной среды
uv venv

# Активация виртуальной среды
. .venv/bin/activate
# .venv\Scripts\activate на Windows
```

> [!note] Глоссарий
> *pre-commit хуки* — это скрипты, которые автоматически запускаются перед коммитом и проверяют/исправляют код (линтеры, форматирование и т.п.), чтобы в репозиторий попадал только корректный код.

```bash
# Установка pre-commit хуков
uv run pre-commit install
```

```bash
# Установка зависимостей
uv sync
```

## 3. Запусти векторную базу Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 4. Структура проекта

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

## Multimodal RAG Pipeline for Medical Imaging

```mermaid
flowchart TD

%% ----------- INPUT ----------------
A[User Uploads Image<br/>+ Optional Question] --> B[FastAPI Backend]

%% ----------- VISION MODEL ---------
B --> C[Vision Model (Model-1)<br/>Qwen2-VL / Qwen3-VL / MedCLIP]
C --> D[Image Caption:<br/>"Preliminary medical description"]

%% ----------- QUERY PREPARATION ----
D --> E[Embeddings Model<br/>SentenceTransformers]
E --> F[Query Vector]

%% ----------- VECTOR SEARCH --------
F --> G[Qdrant Vector DB]
G --> H[Top-K Similar Medical Cases<br/>Descriptions + Metadata]

%% ----------- PROMPT BUILD ---------
H --> I[Build RAG Prompt<br/>caption + retrieved context + question]

%% ----------- LLM ------------------
I --> J[LLM (Model-2)<br/>Local or Remote GPU<br/>Qwen3-VL-32B / LLaMA / DeepSeek]

%% ----------- OUTPUT ---------------
J --> K[Final Answer:<br/>Medical reasoning, uncertainty, next steps]
K --> L[Return Response to User<br/>+ Sources / Similar Cases]
```
