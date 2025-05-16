#!/bin/bash

# Check if a text file is provided as an argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <text_file> [output_csv] [prompt_type]"
    exit 1
fi

PYTHON_CMD="python3 src/cottl.py"
OUTPUT_CSV="${2:-output_results.csv}"
PROMPT_TYPE="${3:-picknplace}"

echo "Using prompt type: $PROMPT_TYPE"

# Create the output CSV file with headers
echo "Input Text,intermediate_output,Output LTL,Label,Match,Explanation" > "$OUTPUT_CSV"

# Initialize counters
total_matches=0
total_processed=0

# Read and process the text file line by line
while IFS=';' read -r text label || [[ -n "$text" ]]; do
    echo "Processing input Text: $text"
    total_processed=$((total_processed + 1))

    # Execute the Python script and capture its output
    if ! output_ltl=$($PYTHON_CMD --model gpt4o --keyfile keys/gpt.txt --plan_ins "$text" --num_tries 2 --temperature 0.1 --prompt "$PROMPT_TYPE"); then
        echo "Error running Python script for: $text"
        echo "\"$text\",\"Error processing\",\"Error\",\"$label\",\"0\",\"\"" >> "$OUTPUT_CSV"
        continue
    fi

    # Extract the intermediate components
    intermediate_output=$(echo "$output_ltl" | sed -n '/Intermediate components:/,/==================================================$/p' | 
                         grep -v "=\{10,\}" | tr -d '\n' | sed 's/[[:space:]]\+/ /g')
    
    # Get ONLY the formula value (extract text after "Formula: ")
    formula=$(echo "$output_ltl" | grep "Formula:" | sed 's/^Formula: //')
    
    # Compare the formula with the label
    label_removed=$(echo "$label" | sed -e 's/<loc>//g' -e 's/[[:space:]]*//g' -e 's/<\/loc>//g')
    formula_removed=$(echo "$formula" | sed -e 's/<loc>//g' -e 's/[[:space:]]*//g' -e 's/<\/loc>//g')
    
    # For debugging
    echo "Extracted formula: '$formula'"
    echo "Expected label: '$label'"
    
    # Check if formula matches label
    if [ "$formula_removed" = "$label_removed" ]; then
        match=1
        total_matches=$((total_matches + 1))
    else
        match=0
    fi
    
    # Properly escape CSV values
    text_escaped="${text//\"/\"\"}"
    intermediate_escaped="${intermediate_output//\"/\"\"}"
    formula_escaped="${formula//\"/\"\"}"
    label_escaped="${label//\"/\"\"}"
    
    # Appending output CSV file
    echo "\"$text_escaped\",\"$intermediate_escaped\",\"$formula_escaped\",\"$label_escaped\",\"$match\",\"\"" >> "$OUTPUT_CSV"
done < "$1"

echo "Total matches: $total_matches out of $total_processed"