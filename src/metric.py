import pandas as pd
import spot
import argparse
import sys
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate LTL formula equivalence")
    parser.add_argument("--input", required=True, help="Input CSV file containing LTL formulas")
    parser.add_argument("--output", required=True, help="Output CSV file for results")
    args = parser.parse_args()

    print(f"Reading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    print(f"Processing {len(df)} rows...")

    equivalence_results = []
    equivalence_numeric = []
    
    # Process each row with a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df)):
        formula1 = row['Output LTL']
        formula2 = row['Label']

        # Skip if either formula is empty
        if pd.isna(formula1) or pd.isna(formula2) or formula1 == '' or formula2 == '':
            print(f"Warning: Empty formula in row {index}")
            equivalence_results.append("Missing formula")
            equivalence_numeric.append(-1)  # Use -1 to indicate missing formula
            continue

        try:
            # Try to convert string formulas to spot formulas
            spot_formula1 = spot.formula(str(formula1))
            spot_formula2 = spot.formula(str(formula2))
            are_eq = spot.are_equivalent(spot_formula1, spot_formula2)
            equivalence_results.append("Equivalent" if are_eq else "Not equivalent")
            equivalence_numeric.append(1 if are_eq else 0)
        except Exception as e:
            print(f"Error for row {index}: {e}")
            equivalence_results.append("Error")
            equivalence_numeric.append(-1)  # -1 indicate an error

    # Add results to DataFrame
    df['Equivalence Result'] = equivalence_results
    df['Equivalence Numeric'] = equivalence_numeric

    # Calculate and display summary
    valid_results = [x for x in equivalence_numeric if x >= 0]
    eq_count = sum(1 for x in valid_results if x == 1)
    total_valid = len(valid_results)
    print(f"\nResults: {eq_count} equivalent out of {total_valid} valid formulas " +
          f"({eq_count/total_valid*100:.2f}% if valid, {eq_count/len(df)*100:.2f}% of total)")
    
    # Save the modified DataFrame
    print(f"Saving results to {args.output}")
    df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
