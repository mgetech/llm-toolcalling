#!/usr/bin/env python3
import argparse

import datasets
from openai import OpenAI

# NOTE: This is just a skeleton function. You should implement the actual tool calling logic here.
# Feel free to change whatever you like; just the CLI and expected output format are important.


def process_dataset(
    ds: datasets.Dataset, model: str, base_url: str, api_key: str
) -> datasets.Dataset:
    """Skeleton function for processing the dataset and adding the tool calls."""

    # Initialize the OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)

    my_answers = []
    # Call the Arcee-Agent Model you hosted to create the tool calls; you will have to call once for each row in the dataset
    for query, tools in zip(ds["query"], ds["tools"]):
        try:
            # this should work if you set up the inference server correctly
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Test for query: {query}",
                    }
                ],
                model=model,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            # Only for testing purposes, there shouldn't be any errors if you set up the inference server and called the script correctly
            print(f"Error calling the model: {e}")
            answer = f"Error for query: {query}"

        # REPLACE THIS WITH YOUR ACTUAL PARSING LOGIC
        formatted_answer = answer
        my_answers.append(formatted_answer)

    # you should add the new column 'my_answers' with the expected tool calls to the dataset
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
