# llm-toolcalling
## 🧠 Project Overview

This project is a Python CLI that uses a locally-hosted LLM (**Arcee-Agent**) via **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** to generate structured tool calls based on user queries. The tool calls are formatted as JSON to match the structure required by the challenge.

---

## 🚀 How It Works

- Loads queries from the _dev_ dataset in `dataset/`
- Sends each query (along with tool descriptions) to the local LLM running with `llama-cpp-python`
- Parses the model's output to extract:
    - the correct tool(s) to call
    - only the relevant arguments, properly formatted
- Appends the results to the dataset in a new column: `"my_answers"`
- Saves the output in `my_dataset/` for evaluation
