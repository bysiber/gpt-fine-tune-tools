"""
This script prepares data for fine-tuning GPT models using the OpenAI API.

It takes a directory of text files representing desired assistant responses 
and automatically generates plausible user queries to create a complete 
training dataset. This allows you to fine-tune a GPT model to 
mimic a specific writing style or persona.

Usage:
  python prepare_gpt_data.py \
    --persona "<Your desired system persona>" \
    --data_dir "<Directory with your text files>" \
    --output_file "<Output file for training data>"
"""


import os
import json
import openai
from packaging import version

# Ensure OpenAI library version compatibility
MINIMUM_OPENAI_VERSION = "1.1.1"
if version.parse(openai.__version__) < version.parse(MINIMUM_OPENAI_VERSION):
    raise ValueError(
        f"OpenAI version {openai.__version__} is incompatible. "
        f"Version {MINIMUM_OPENAI_VERSION} or higher is required."
    )

class FineTuningDataPreprocessor:
    """
    Prepares training data for GPT fine-tuning from a directory of text files.
    Each file is treated as a separate conversation turn by the assistant.
    """

    def __init__(self, persona: str, data_directory: str):
        """
        Initializes the data preprocessor.

        Args:
            persona: The system persona/prompt to use for the fine-tuning.
            data_directory: Path to the directory containing the text files.
        """
        self.persona = persona
        self.data_directory = data_directory
        self.client = openai.OpenAI()

    def _generate_simulated_query(self, assistant_response: str) -> str:
        """
        Uses the OpenAI API to generate a plausible user query 
        that could have led to the given assistant response.

        Args:
            assistant_response: The assistant's response text.

        Returns:
            The generated user query.
        """
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a query generator. Given a piece of text, craft a user query that might have resulted in that text as an output.",
                },
                {"role": "user", "content": assistant_response},
            ],
        )
        return response.choices[0].message.content.strip()

    def _process_text_file(self, file_path: str) -> dict:
        """
        Processes a single text file into a training data entry.

        Args:
            file_path: Path to the text file.

        Returns:
            A dictionary representing a single training data entry.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            assistant_response = file.read()
        
        simulated_query = self._generate_simulated_query(assistant_response)
        return {
            "messages": [
                {"role": "system", "content": self.persona},
                {"role": "user", "content": simulated_query},
                {"role": "assistant", "content": assistant_response},
            ]
        }

    def generate_training_data(self) -> list:
        """
        Processes all text files in the data directory 
        and generates the complete training data.

        Returns:
            A list of training data entries.
        """
        training_data = []
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".txt"): 
                file_path = os.path.join(self.data_directory, filename)
                data_entry = self._process_text_file(file_path)
                training_data.append(data_entry)
        return training_data

def main():
    """Parses command-line arguments and executes the data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare data for GPT fine-tuning."
    )
    parser.add_argument(
        "--persona", required=True, help="The system persona for the model."
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing the text files."
    )
    parser.add_argument(
        "--output_file", required=True, help="Output file for the training data."
    )
    args = parser.parse_args()

    preprocessor = FineTuningDataPreprocessor(args.persona, args.data_dir)
    training_data = preprocessor.generate_training_data()

    with open(args.output_file, "w", encoding="utf-8") as f:
        for data_entry in training_data:
            json.dump(data_entry, f)
            f.write("\n")

if __name__ == "__main__":
    main()
