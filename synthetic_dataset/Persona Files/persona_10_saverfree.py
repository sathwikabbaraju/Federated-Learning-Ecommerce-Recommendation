
import pandas as pd
import os

def simulate_saverfree_behavior(file_path: str, output_path: str):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.drop(columns=['image', 'link'], inplace=True)

    result_rows = []

    for category, group in df.groupby('main_category'):
        filtered = g.assign(discount_percent=(g['actual_price'] - g['discount_price']) / g['actual_price']).sort_values(by='discount_percent', ascending=False)
        if not filtered.empty:
            selected = filtered.iloc[0]
            result_rows.append(selected)

    result_df = pd.DataFrame(result_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Generated saverfree synthetic dataset at: {output_path}")

# Example usage:
# simulate_saverfree_behavior('path/to/input.csv', 'synthetic_data/saverfree.csv')
