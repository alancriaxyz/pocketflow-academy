# LLM Wrapper Example

This example demonstrates how to create and use an LLM wrapper with PocketFlow, including:

1. Basic LLM calls with proper error handling
2. Chat history management
3. Token counting and context management
4. Caching for efficiency
5. Structured output handling

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the example:
```bash
python main.py
```

This will demonstrate:
- Basic LLM completion
- Chat conversation with history
- Structured output generation
- Caching and token management
- Error handling and retries 