# Запуск проекта

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
