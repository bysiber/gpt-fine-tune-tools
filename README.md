# gpt-fine-tune
 GPT Fine-Tuning Data Preparation with `prepare_gpt_data.py`

This repository provides tools and examples for fine-tuning GPT models. This script, `prepare_gpt_data.py`, is designed to streamline the data preparation process for fine-tuning by automatically generating user queries.

## `prepare_gpt_data.py`: Generating User Queries for Training Data

The `prepare_gpt_data.py` script simplifies the creation of fine-tuning datasets for GPT models. 

**Purpose:**

- Takes a set of desired assistant responses (your text files).
- Automatically generates plausible user queries that could have led to those responses, using a GPT model.
- Structures the data into the correct JSONL format for OpenAI's fine-tuning API.

**This is especially useful when:**

- You want to fine-tune a GPT model to mimic a specific writing style or persona based on example outputs.
- You have a collection of ideal assistant responses but need to generate corresponding user inputs.

**Usage:**

1. **Prepare your data:** 
   - Create a directory containing `.txt` files.
   - Each file should hold a single example of your desired assistant output.

2. **Run the script:**

   ```bash
   python prepare_gpt_data.py \
     --persona "Your desired system persona" \
     --data_dir "path/to/your/text/files" \
     --output_file "training_data.jsonl"
