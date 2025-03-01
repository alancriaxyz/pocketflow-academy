# PocketFlow Text Converter

A practical example demonstrating how to use PocketFlow to create an interactive text converter. This example showcases important concepts like data flow between nodes, user choice-based branching, and state management using the shared store.

## Features

- Convert text to UPPERCASE
- Convert text to lowercase
- Reverse text
- Remove extra spaces
- Interactive command-line interface
- Continuous flow with option to process multiple texts

## Project Structure

```
.
├── flow.py        # Nodes and flow implementation
├── main.py        # Application entry point
└── README.md      # Documentation
```

## Implementation Details

1. **TextInput Node**:
   - `prep()`: Gets text input from user
   - `post()`: Shows options menu and returns action based on choice

2. **TextTransform Node**:
   - `prep()`: Gets text and choice from shared store
   - `exec()`: Applies the chosen transformation
   - `post()`: Shows result and asks if continue

3. **Flow Structure**:
   - Input → Transform → (loop back to Input or exit)
   - Demonstrates branching based on actions ("transform", "input", "exit")

## How to Run

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the example:
```bash
python main.py
```

## What You'll Learn

This example demonstrates several important PocketFlow concepts:

- **Node Architecture**: How to structure logic using prep/exec/post pattern
- **Flow Control**: How to use actions to control flow between nodes
- **Shared Store**: How to share data between nodes
- **Interactivity**: How to create interactive flows with user input
- **Branching**: How to implement different paths based on choices

## Additional Resources

- [PocketFlow Documentation](https://the-pocket.github.io/PocketFlow/)
- [Flow Guide](https://the-pocket.github.io/PocketFlow/flow.html)
- [Node Guide](https://the-pocket.github.io/PocketFlow/node.html) 