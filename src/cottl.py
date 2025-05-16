import argument
import llm
import os
import sys
from pathlib import Path


def setup_environment():
    """Setup necessary directories and files"""
    required_dirs = ["prompts", "keys"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Warning: Could not create {dir_path}: {e}")
    
    # Create README in keys directory if it doesn't exist
    keys_readme = Path("keys") / "README.md"
    if not keys_readme.exists():
        try:
            with open(keys_readme, 'w') as f:
                f.write("# API Keys Directory\n\n"
                        "Place your API keys in this directory:\n\n"
                        "1. `hf_key.txt` - HuggingFace API key\n"
                        "2. `gpt.txt` - OpenAI API key\n")
            print(f"Created {keys_readme}")
        except Exception as e:
            print(f"Warning: Could not create {keys_readme}: {e}")

def main():

    setup_environment()
    
    # Parse arguments
    args = argument.parse_args()
    
    try:
        # Process the request
        result = llm.call(args)
        
        # Output the results
        if result and isinstance(result, tuple) and len(result) >= 2:
            ltl_translation, intermediate_data = result
            
            print("\n" + "="*50)
            print("LTL TRANSLATION RESULT:")
            print("="*50)
            print(f"Formula: {ltl_translation[0]}")
            print(f"Confidence: {ltl_translation[1]:.2f}")
            
            if intermediate_data and len(intermediate_data) >= 4:
                print("\nIntermediate components:")
                for i, key in enumerate(intermediate_data[0]):
                    if i < len(intermediate_data[1]) and intermediate_data[1][i]:
                        print(f"  - {key}: {intermediate_data[1][i][0]}")
            print("="*50)
        else:
            print("Error: Unexpected result format")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files exist.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
