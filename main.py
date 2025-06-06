#!/usr/bin/env python3
import argparse

import datasets
from openai import OpenAI
import json


def process_dataset(
    ds: datasets.Dataset, model: str, base_url: str, api_key: str
) -> datasets.Dataset:
    """Skeleton function for processing the dataset and adding the tool calls."""
    # Initialize the OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)

    my_answers = []
    # Call the Arcee-Agent Model 
    for query, tools in zip(ds["query"], ds["tools"]):
        try:
            prompt = f"""You have these tools:
    {tools}

    Now, for this user request:
    {query}

    Reply with a JSON _array_ of tool-call objects. Each object must have:
      • "name": the tool’s name  
      • "arguments": an object with only the parameters needed for that specific call  

    If the request requires multiple calls (even to the same tool with different arguments), include one object per needed call.
    Do not output unused parameters or any extra text.
    """
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
            )
            answer = response.choices[0].message.content.strip()

            # Attempt to parse response as JSON
            try:
                parsed = json.loads(answer)
                formatted_answer = json.dumps(parsed)
            except Exception as parse_err:
                print(f"[ParseError] Could not parse model response as JSON:\n{answer}")
                formatted_answer = "[]"

        except Exception as e:
            print(f"[ModelError] Failed on query:\n{query}\nError: {e}")
            formatted_answer = "[]"

        my_answers.append(formatted_answer)

    return ds.add_column("my_answers", my_answers)


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Generate tool calls using an LLM")
    parser.add_argument("--model", required=True, help="Name of the model to use")
    parser.add_argument(
        "--base_url",
        required=True,
        help="Base URL of the inference server, e.g. http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api_key", required=True, help="API key for the inference server"
    )
    args = parser.parse_args()

    ds = datasets.load_from_disk("./dataset/")
    assert isinstance(ds, datasets.Dataset)
    # Process the dataset and generate tool calls
    submission_ds = process_dataset(ds, args.model, args.base_url, args.api_key)
    # Save the resulting dataset
    submission_ds.save_to_disk("./my_dataset")


if __name__ == "__main__":
    main()
