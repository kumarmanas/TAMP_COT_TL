# CoT-TL: Chain-of-Thought Temporal Logic Translation

This tool translates natural language instructions to Linear Temporal Logic (LTL) formulas using large language models.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up API keys:
   - For HuggingFace models: Create a file `keys/hf_key.txt` with your HuggingFace API key
   - For OpenAI models: Create a file `keys/gpt.txt` with your OpenAI API key

3. Install Spot library:
   - Follow instructions at https://spot.lre.epita.fr/install.html

## Semantic Role Labeling

You have several options to implement semantic role labeling:
- Use the VerbNet library from [https://github.com/cu-clear/verbnet](https://github.com/cu-clear/verbnet)
- Use NLTK library by importing VerbNet thematic roles with `from nltk.corpus import verbnet` and generate thematic roles for each planning instruction
- Include semantic role labeling as part of the prompt (recommended), as LLMs are able to understand semantic role labeling from examples in the prompt

## Prompting

Provide prompt files in the `/prompts` directory:
- `drone.txt` - prompt used for drone planning
- Other prompt files can be added as needed

## Running the Code

### Single Inference
For single inference, use:
```bash
python3 src/cottl.py --model gpt4 --num_tries 5 --temperature 0.2 --prompt picknplace --plan_ins "move back and forth to the basket and pick up any blocks in your path and place them in the basket." --keyfile your_key_file.txt
```

We use the Hugging Face Inference API for generation. Local model inference is also provided and can be customized to add newer models like DeepSeek.

### Batch Processing
For batch processing and metric generation, we use a bash script that stores the final LTL output (skipping explainable output). You can customize the script for your specific files and requirements.

Use our provided bash script `train.sh` which requires an additional input flag for the dataset path. Note that our dataset is in `.txt` format where text and LTL are separated by `;`.

To process in batches:
```bash
chmod +x src/train.sh
./src/train.sh src/datasets/dataset_name.txt results.csv prompt_name
```

Example:
```bash
./src/train.sh src/datasets/clean_up.txt results.csv cleanup
```

Then generate metrics with:
```bash
python3 src/metric.py --input results.csv --output evaluated_results.csv
```

This checks how many formulas are correctly evaluated via Spot verification.

### Input File Format
```
Natural language instruction 1;Expected LTL formula 1
Natural language instruction 2;Expected LTL formula 2
...
```

## Dataset Evaluation Metric Generation
We use the `metric.py` script to generate the total number of correct LTL formulas in the dataset. This script also handles extra open and close brackets that might be introduced during training.

## Notes
1. You need to manually add API keys. Support for spot feedback will be updated after the patent application.
2. See attached video for the complete workflow demonstrating our interpretable output and drone planning.

## Acknowledgements
CoT-TL is greatly inspired by the following outstanding contributions to the open-source community:
- nl2spec (https://github.com/realChrisHahn2/nl2spec/)
- ToT (https://github.com/princeton-nlp/tree-of-thought-llm)

## Citation
If you find our code and paper can help, please cite our paper as:

```bibtex
@INPROCEEDINGS{10801817,
  author={Manas, Kumar and Zwicklbauer, Stefan and Paschke, Adrian},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={CoT-TL: Low-Resource Temporal Knowledge Representation of Planning Instructions Using Chain-of-Thought Reasoning}, 
  year={2024},
  doi={10.1109/IROS58592.2024.10801817}
}
```
