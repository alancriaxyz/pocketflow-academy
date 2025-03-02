# Embeddings Tool Example

This example demonstrates how to use embeddings effectively with PocketFlow, including:

1. Basic text embedding generation
2. Semantic search using embeddings
3. Document similarity comparison
4. Caching and batch processing
5. Best practices for production use

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
- Creating embeddings for sample texts
- Finding similar documents using semantic search
- Comparing document similarities
- Efficient batch processing of multiple texts 