# PocketFlow Academy

Welcome to PocketFlow Academy! 👋 Here you'll find practical examples and tutorials to help you learn [PocketFlow](https://github.com/The-Pocket/PocketFlow).

## Examples

### Core Concepts
#### Node
- [`pocketflow-node`](./pocketflow-node) - Learn Node concepts through a practical example demonstrating prep->exec->post lifecycle, error handling, and retries

#### Flow
- [`pocketflow-flow`](./pocketflow-flow) - Learn Flow concepts through a practical example demonstrating action-based transitions and branching

#### Communication
- [`pocketflow-communication`](./pocketflow-communication) - A word counter app showcasing the Shared Store pattern for communication between nodes

#### BatchNode
- [`pocketflow-batch-node`](./pocketflow-batch-node) - Learn BatchNode through a CSV processor that handles large files in chunks

#### BatchFlow
- [`pocketflow-batch-flow`](./pocketflow-batch-flow) - Learn BatchFlow through an image processor that applies multiple filters to multiple images

### Advanced
#### Nested BatchFlow
- [`pocketflow-nested-batch`](./pocketflow-nested-batch) - Learn Nested BatchFlow through a school grades calculator that processes multiple classes with multiple students

#### AsyncNode
- [`pocketflow-async-basic`](./pocketflow-async-basic) - A simple recipe finder that demonstrates asynchronous operations. The example shows how to:
  - Handle user input without blocking
  - Make async API calls
  - Process results with async LLM calls
  - Chain async operations in a flow

#### Parallel Processing
- [`pocketflow-parallel-batch-node`](./pocketflow-parallel-batch-node) - Learn how to process multiple items concurrently using ParallelBatchNode
- [`pocketflow-parallel-batch-flow`](./pocketflow-parallel-batch-flow) - Advanced example of parallel processing with multiple flows running concurrently

### Tools and Utilities
- [`pocketflow-tool-embeddings`](./pocketflow-tool-embeddings) - Example of how to integrate and use OpenAI embeddings with proper environment configuration and code organization

### Getting Started
- [`pocketflow-hello-world`](./pocketflow-hello-world) - Build your first complete PocketFlow project with step-by-step guidance

## What This Example Demonstrates

- Basic PocketFlow setup in a Python project
- Standard project structure
- Testing setup
- Best practices for Python development with PocketFlow
- Error handling and retry mechanisms
- Working with LLMs in PocketFlow
- Environment configuration and security best practices
- Parallel processing and performance optimization

## Additional Resources

- [PocketFlow Documentation](https://the-pocket.github.io/PocketFlow/)

## Contributing

Want to contribute? Great! Feel free to submit a pull request with your own example. Make sure to:
1. Include comprehensive documentation
2. Follow Python best practices
3. Add appropriate tests

## License

This repository is licensed under the same terms as PocketFlow. See [LICENSE](LICENSE) for more details. 