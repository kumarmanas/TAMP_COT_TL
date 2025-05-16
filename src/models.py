import os
import json
import requests
import ast
import sys
from pathlib import Path
from functools import lru_cache
import spot
import calculation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Import OpenAI with conditional handling for both old and new APIs
try:
    from openai import OpenAI  # New client-based API
    OPENAI_NEW_API = True
except ImportError:
    import openai
    OPENAI_NEW_API = False

# Helper constants
MODEL_ENDPOINTS = {
    "deepseek": "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistral": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "starcoder": "https://api-inference.huggingface.co/models/bigcode/starcoder"
}

DEFAULT_PROMPT = """I need to translate natural language instructions into Linear Temporal Logic (LTL) formulas.

Let me work through this step by step:

1. First, I'll identify the key actions and objects in the instruction.
2. Then, I'll determine temporal relationships between actions.
3. Finally, I'll formalize this as an LTL formula using operators like:
   - F (eventually)
   - G (globally/always)
   - X (next)
   - U (until)
   - & (and)
   - | (or)
   - ! (not)
   - -> (implies)

Let me give a detailed explanation of my translation process."""

def locate_api_key(args, filename):
    """Locate API key file with improved search logic"""
    # Explicit path provided
    if args.keyfile:
        return args.keyfile
    
    # Key directory + filename
    if args.keydir:
        key_path = Path(args.keydir) / filename
        if key_path.exists():
            return str(key_path)
    
    # Search in common locations
    search_paths = [
        Path("keys") / filename,
        Path(__file__).parent.parent / "keys" / filename,
        Path.home() / ".keys" / filename
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"Found API key at: {path}")
            return str(path)
    
    # Not found
    print(f"Error: Could not locate {filename}")
    print(f"Searched in: {', '.join(str(p) for p in search_paths)}")
    raise FileNotFoundError(f"API key file {filename} not found")

def read_key_file(path):
    """Read API key from file, with better error handling"""
    try:
        with open(path, 'r') as f:
            content = f.read().strip()
            
        # Check if content is a comment
        if content.startswith('#'):
            print(f"Warning: The file {path} contains a comment, not an API key.")
            print("Please replace the content with your actual API key.")
            raise ValueError(f"No valid API key in {path}")
            
        if not content:
            raise ValueError(f"Empty API key file: {path}")
        
        return content
    except Exception as e:
        print(f"Error reading API key from {path}: {e}")
        raise

def get_api_key(args, default_filename=None):
    """Get API key with improved error handling"""
    key_path = locate_api_key(args, default_filename)
    return read_key_file(key_path)

def find_prompt_template(args):
    """Find the appropriate prompt template file or return default"""
    # Find prompt directory
    prompt_dirs = [
        Path("prompts"),
        Path(__file__).parent.parent / "prompts",
        Path(__file__).parent / "prompts"
    ]
    
    prompt_dir = None
    for path in prompt_dirs:
        if path.is_dir():
            prompt_dir = path
            print(f"Using prompt directory: {path}")
            print(os.getcwd())
            print(prompt_dir)
            break
    
    if prompt_dir is None:
        # Try to create prompts directory
        try:
            Path("prompts").mkdir(exist_ok=True)
            prompt_dir = Path("prompts")
            print(f"Created prompt directory: {prompt_dir}")
        except Exception as e:
            print(f"Warning: Could not create prompts directory: {e}")
            return DEFAULT_PROMPT
    
    # Map prompt name to file
    prompt_files = {
        "drone_processed": prompt_dir / "drone_processed.txt",
        "picknplace": prompt_dir / "picknplace.txt",
        "cleanup": prompt_dir / "cleanup.txt"
    }
    
    # If prompt name matches one of our templates
    if args.prompt in prompt_files:
        prompt_path = prompt_files[args.prompt]
        print(f"Looking for prompt file: {prompt_path} (exists: {prompt_path.exists()})")
        if prompt_path.exists():
            try:
                with open(prompt_path, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading prompt file {prompt_path}: {e}")
                return DEFAULT_PROMPT
        else:
            print(f"Prompt file not found: {prompt_path}")
            # If picknplace.txt is missing, try to create it
            if args.prompt == "picknplace":
                try:
                    create_picknplace_prompt(prompt_dir)
                    if prompt_path.exists():
                        with open(prompt_path, 'r') as f:
                            return f.read()
                except Exception as e:
                    print(f"Error creating picknplace prompt: {e}")
            
            return DEFAULT_PROMPT
    
    # If the prompt is not a template name, use it directly
    return args.prompt

def create_picknplace_prompt(prompt_dir):
    """Create the picknplace prompt file"""
    picknplace_content = """I need to translate a natural language instruction involving pick-and-place actions into Linear Temporal Logic (LTL) formulas.

When translating, I'll follow these steps:
1. Identify the key actions in the instruction: pick(object), place(object, location), move(location)
2. Determine the sequence and conditions for these actions
3. Express this using LTL operators:
   - F (eventually): something must happen in the future
   - G (globally/always): something must be true at all times
   - X (next): something must happen in the next state
   - U (until): something must be true until another condition becomes true
   - & (and), | (or), ! (not), -> (implies): standard logical operators

I'll focus on these basic LTL patterns for pick-and-place tasks:
- Sequential actions: first A, then B = F(A & F(B))
- Conditional actions: if condition C, then action A = G(C -> F(A))
- Safety properties: never do A = G(!A)
- Fairness properties: always eventually do A = GF(A)

Let me translate the given instruction by breaking it down into simpler parts, explaining each component, and then building the complete LTL formula."""
    
    with open(prompt_dir / "picknplace.txt", 'w') as f:
        f.write(picknplace_content)
    print(f"Created picknplace.txt prompt file")

def construct_prompt(template, instruction, given_translations=""):
    """Constructs the final prompt using a template-based approach"""
    # Format the instruction part
    instruction_section = f"Natural Language: {instruction}"
    
    # Format the translations part
    if given_translations and given_translations.strip():
        translations_section = f"Given translations: {given_translations}"
    else:
        translations_section = "Given translations: {}"
    
    # Combine all parts with proper spacing
    sections = [
        template.strip(),
        instruction_section,
        translations_section,
        "Explanation:"
    ]
    
    final_prompt = "\n\n".join(sections)
    
    print("PROMPT STRUCTURE:")
    print(f"- Template: {len(template.strip())} chars")
    print(f"- Instruction: {len(instruction)} chars")
    print(f"- Given translations: {'Custom' if given_translations else 'Empty'}")
    
    return final_prompt

def prompt(args):
    """Generate the prompt using a restructured approach"""
    # Find appropriate template
    template = find_prompt_template(args)
    
    # Build prompt using helper function with different structure
    final_prompt = construct_prompt(
        template=template,
        instruction=args.plan_ins,
        given_translations=args.given_translations if hasattr(args, 'given_translations') else ""
    )
    
    return final_prompt

def parse_formulas(choices):
    parsed_result_formulas = []
    for c in choices:
        try:
            if not c or not isinstance(c, str):
                continue
                
            # Try various ways of identifying the formula
            formula_str = None
            
            # Original method
            formula_parts = c.split("So the final LTL translation is:")
            if len(formula_parts) >= 2:
                formula_str = formula_parts[1].strip().strip(".")
            
            # Alternative patterns to find the formula
            if not formula_str:
                patterns = [
                    "final LTL formula is:", 
                    "LTL formula:", 
                    "final formula:", 
                    "translated formula is:", 
                    "formula is:",
                    "Complete LTL formula:"
                ]
                
                for pattern in patterns:
                    if pattern in c:
                        parts = c.split(pattern)
                        if len(parts) >= 2:
                            # Extract text after the pattern, up to the next period or paragraph
                            potential_formula = parts[1].strip().split('.')[0].split('\n')[0]
                            if potential_formula:
                                formula_str = potential_formula
                                break
            
            # Last resort - try to find anything that looks like an LTL formula
            if not formula_str and ("G(" in c or "F(" in c or " U " in c):
                import re
                # Look for patterns like G(...), F(...), etc.
                matches = re.findall(r'[GFX]\([^)]+\)| U |&|\||\-\>|!', c)
                if matches:
                    # Find the paragraph with the most LTL operators
                    paragraphs = c.split('\n\n')
                    best_paragraph = max(paragraphs, key=lambda p: sum(1 for m in matches if m in p))
                    formula_str = best_paragraph.strip()
            
            if formula_str:
                # Clean the formula string and parse it
                formula_str = formula_str.strip()
                parsed_formula = spot.formula(formula_str)
                parsed_result_formulas.append(parsed_formula)
                
        except Exception as e:
            print(f"Error parsing formula: {e}")
            # Still include the string version if parsing fails
            if 'formula_str' in locals() and formula_str:
                parsed_result_formulas.append(formula_str)
    
    return parsed_result_formulas

def parse_explanation_dictionary(choices, plan_ins):
    """Parse explanation dictionaries from model outputs"""
    parsed_explanation_results = []
    for c in choices:
        if not c or not isinstance(c, str):
            continue
        try:
            if "dictionary" not in c:
                continue
            dict_parts = c.split("dictionary")
            if len(dict_parts) < 2:
                continue
            dict_str = dict_parts[1].split("{", 1)
            if len(dict_str) < 2:
                continue
            dict_content = dict_str[1].split("}", 1)[0]
            dict_string = "{" + dict_content + "}"
            parsed_dict = ast.literal_eval(dict_string)
            parsed_dict = dict(filter(lambda x: x[0] != plan_ins, parsed_dict.items()))
            if parsed_dict:
                parsed_explanation_results.append(parsed_dict)
        except Exception as e:
            print(f"Error parsing explanation dictionary: {e}")
    return parsed_explanation_results

def extract_subinfo(choices, args, n):
    """Extract and process information from model outputs"""
    parsed_result_formulas = parse_formulas(choices)
    print("Results of multiple runs:")
    print(parsed_result_formulas)
    
    final_translation = calculation.ambiguity_final_translation(parsed_result_formulas, n)
    parse_explain = parse_explanation_dictionary(choices, args.plan_ins)
    
    # Handle locked translations if provided
    locked_trans = {}
    if hasattr(args, 'locked_translations') and args.locked_translations:
        try:
            locked_trans = ast.literal_eval(args.locked_translations)
        except:
            print("Warning: Could not parse locked translations")
    
    # Process intermediate translations with a different approach
    intermediate_translation = calculation.ambiguity_detection_translations(
        parse_explain, n, locked_trans
    )
    
    # Use the restructured transform_translations function
    intermediate_output = calculation.transform_translations(intermediate_translation)
    return final_translation, intermediate_output

# Model implementations
def deepseek(args):
    """Implementation for deepseek model"""
    n = args.num_tries
    input_prompt = prompt(args)
    API_URL = MODEL_ENDPOINTS["deepseek"]
    key = get_api_key(args, "hf_key.txt")
    headers = {"Authorization": f"Bearer {key}"}
    
    choices = []
    for i in range(n):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={
                    "inputs": input_prompt,
                    "options": {"use_cache": False, "wait_for_model": True},
                    "parameters": {
                        "return_full_text": False,
                        "do_sample": False,
                        "max_new_tokens": 500,
                        "temperature": args.temperature,
                    },
                },
                timeout=60 
            )
            
            if response.status_code != 200:
                print(f"Error from API: {response.status_code} - {response.text}")
                continue
                
            raw_output = response.json()
            
 
            if isinstance(raw_output, dict) and 'error' in raw_output:
                print(f"Error from API: {raw_output['error']}")
                continue
            

            if not isinstance(raw_output, list) or len(raw_output) == 0 or "generated_text" not in raw_output[0]:
                print("Unexpected response format:", raw_output)
                continue
            
            output = raw_output[0]["generated_text"].split("FINISH")[0]
            choices.append(output)
        except Exception as e:
            print(f"Error in API call: {e}")
    
    # Check if we got any valid responses
    if not choices:
        print("Warning: No valid responses received")
        return "T", [[], [], [], []]
    
    return extract_subinfo(choices, args, len(choices))

def gpt4(args):
    """Implementation for GPT-4 model"""
    key = get_api_key(args, "gpt.txt")
    n = min(int(args.num_tries) if args.num_tries else 3, 5)
    input_prompt = prompt(args)
    
    choices = []
    try:
        if OPENAI_NEW_API:
            # New OpenAI API
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": input_prompt}],
                n=n,
                temperature=args.temperature,
                stop=["FINISH"]
            )
            
            for choice in response.choices:
                output = choice.message.content
                choices.append(output)
                
        else:
            # Old OpenAI API
            openai.api_key = key
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": input_prompt}],
                n=n,
                temperature=args.temperature,
                stop="FINISH",
            )
            
            for choice in response["choices"]:
                output = choice["message"]["content"]
                choices.append(output)
                
        return extract_subinfo(choices, args, len(choices))
    except Exception as e:
        print(f"Error calling GPT-4 API: {e}")
        return "T", [[], [], [], []]

def gpt3(args):
    """Implementation for GPT-3.5 model"""
    key = get_api_key(args, "gpt.txt")
    n = min(int(args.num_tries) if args.num_tries else 3, 5)
    input_prompt = prompt(args)
    
    choices = []
    try:
        if OPENAI_NEW_API:
            # New OpenAI API
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input_prompt}],
                n=n,
                temperature=args.temperature,
                stop=["FINISH"]
            )
            
            for choice in response.choices:
                output = choice.message.content
                choices.append(output)
                
        else:
            # Old OpenAI API
            openai.api_key = key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input_prompt}],
                n=n,
                temperature=args.temperature,
                stop="FINISH",
            )
            
            for choice in response["choices"]:
                output = choice["message"]["content"]
                choices.append(output)
                
        return extract_subinfo(choices, args, len(choices))
    except Exception as e:
        print(f"Error calling GPT-3 API: {e}")
        return "T", [[], [], [], []]

def gpt4o(args):
    """Implementation for GPT-4o model"""
    key = get_api_key(args, "gpt.txt")
    n = min(int(args.num_tries) if args.num_tries else 3, 5)
    input_prompt = prompt(args)
    
    choices = []
    try:
        if OPENAI_NEW_API:
            # New OpenAI API
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4o",  # Changed from gpt-4o-mini to gpt-4o
                messages=[{"role": "user", "content": input_prompt}],
                n=1,  # n parameter may not be supported for all models
                temperature=args.temperature,
                stop=["FINISH"]
            )
            
            for choice in response.choices:
                output = choice.message.content
                choices.append(output)
                
        else:
            # Old OpenAI API
            openai.api_key = key
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": input_prompt}],
                n=1,
                temperature=args.temperature,
                stop="FINISH",
            )
            
            for choice in response["choices"]:
                output = choice["message"]["content"]
                choices.append(output)
                
        return extract_subinfo(choices, args, len(choices))
    except Exception as e:
        print(f"Error calling GPT-4o API: {e}")
        return "T", [[], [], [], []]

def mistral(args):
    """Implementation for Mistral model"""
    n = args.num_tries
    input_prompt = prompt(args)
    API_URL = MODEL_ENDPOINTS["mistral"]
    key = get_api_key(args, "hf_key.txt")
    headers = {"Authorization": f"Bearer {key}"}
    
    choices = []
    for i in range(n):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={
                    "inputs": input_prompt,
                    "options": {"use_cache": False, "wait_for_model": True},
                    "parameters": {
                        "return_full_text": False,
                        "do_sample": True,
                        "max_new_tokens": 500,
                        "temperature": args.temperature,
                    },
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Error from API: {response.status_code} - {response.text}")
                continue
                
            raw_output = response.json()
            
            if isinstance(raw_output, dict) and 'error' in raw_output:
                print(f"Error from API: {raw_output['error']}")
                continue
            
            if not isinstance(raw_output, list) or len(raw_output) == 0 or "generated_text" not in raw_output[0]:
                print("Unexpected response format:", raw_output)
                continue
            
            output = raw_output[0]["generated_text"].split("FINISH")[0]
            choices.append(output)
        except Exception as e:
            print(f"Error in API call: {e}")
    
    if not choices:
        print("Warning: No valid responses received")
        return "T", [[], [], [], []]
    
    return extract_subinfo(choices, args, len(choices))

def starcoder(args):
    """Implementation for Starcoder model"""
    n = args.num_tries
    input_prompt = prompt(args)
    API_URL = MODEL_ENDPOINTS["starcoder"]
    key = get_api_key(args, "hf_key.txt")
    headers = {"Authorization": f"Bearer {key}"}
    
    choices = []
    for i in range(n):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={
                    "inputs": input_prompt,
                    "options": {"use_cache": False, "wait_for_model": True},
                    "parameters": {
                        "return_full_text": False,
                        "do_sample": True,
                        "max_new_tokens": 500,
                        "temperature": args.temperature,
                    },
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Error from API: {response.status_code} - {response.text}")
                continue
                
            raw_output = response.json()
            
            if isinstance(raw_output, dict) and 'error' in raw_output:
                print(f"Error from API: {raw_output['error']}")
                continue
            
            if not isinstance(raw_output, list) or len(raw_output) == 0 or "generated_text" not in raw_output[0]:
                print("Unexpected response format:", raw_output)
                continue
            
            output = raw_output[0]["generated_text"].split("FINISH")[0]
            choices.append(output)
        except Exception as e:
            print(f"Error in API call: {e}")
    
    if not choices:
        print("Warning: No valid responses received")
        return "T", [[], [], [], []]
    
    return extract_subinfo(choices, args, len(choices))

"""Local Model hosting and use for LTL translation"""

MODEL_PATHS = {
    "small": "./models/DeepSeek-R1-Distill-Qwen-1.5B",
    "large": "./models/DeepSeek-R1-Distill-Qwen-32B"
}



@lru_cache(maxsize=2)
def load_deepseek_model(model_size='small'):
    """Load a local DeepSeek model"""
    model_path = MODEL_PATHS.get(model_size, MODEL_PATHS["small"])
    print(f"Loading model from {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model is downloaded to {model_path}")
        raise
        
def deepseek_local(args):
    """Local version of DeepSeek model inference"""
    # Get model_size if specified, default to 'small'
    model_size = 'small'
    if args.model == 'deepseek_large':
        model_size = 'large'
        
    # Add temperature adjustment based on recommendations by the deepseek
    temp = args.temperature
    if temp < 0.5 or temp > 0.7:
        print(f"Warning: Recommended temperature for DeepSeek is 0.5-0.7 (using {temp})")
    
    model, tokenizer = load_deepseek_model(model_size)
    
    # Enforce thinking pattern in prompt
    input_prompt = prompt(args)
    thinking_prompt = "Please start your answer with '<think>\\n' and end your thinking process with '\\n</think>' before giving the final answer."
    enhanced_prompt = f"{input_prompt}\n\n{thinking_prompt}"
    
    # Tokenize input
    inputs = tokenizer(
        enhanced_prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(model.device)
    
    choices = []
    for _ in range(args.num_tries):
        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=temp,
            do_sample=True,  # Enable sampling
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode output
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output.split("FINISH")[0]  # Maintain your existing processing
        choices.append(output)
    
    return extract_subinfo(choices, args, len(choices))