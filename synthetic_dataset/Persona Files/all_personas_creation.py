import pandas as pd
import numpy as np
import os

def persona_datasets(input_file: str, output_folder: str) -> None:
    # Load the cleaned master file
    df = pd.read_csv(input_file)

    # Basic cleaning
    df.drop(columns=['image', 'link'], errors='ignore', inplace=True)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')
    df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
    df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

    # Drop missing critical values
    df.dropna(subset=['rating', 'no_of_ratings', 'discount_price', 'actual_price'], inplace=True)

    # Calculate discount percent and popularity score
    df['discount_percent'] = (df['actual_price'] - df['discount_price']) / df['actual_price']
    df['popularity_score'] = df['rating'] * df['no_of_ratings']

    # Prepare output folder
    os.makedirs(output_folder, exist_ok=True)

    # Persona logic mappings
    personas = {
        "ratermax": lambda g: g[(g['rating'] >= 4.0) & (g['actual_price'] >= np.percentile(g[g['rating'] >= 4.0]['actual_price'], 75))],
        "ratermed": lambda g: g[(g['rating'] >= 3.0) & (g['rating'] < 4.0) & (g['actual_price'] >= np.percentile(g[(g['rating'] >= 3.0) & (g['rating'] < 4.0)]['actual_price'], 75))],
        "impulsivemax": lambda g: g[g['actual_price'] >= np.percentile(g['actual_price'], 75)],
        "valuerater": lambda g: g[(g['rating'] >= 4.0) & (g['discount_percent'].between(0.2, 0.6))],
        "midvalue": lambda g: g[(g['rating'] >= 3.0) & (g['rating'] < 4.0) & (g['discount_percent'].between(0.2, 0.6))],
        "impulsivemid": lambda g: g[g['discount_percent'].between(0.2, 0.6)],
        "middiscount": lambda g: g[(g['rating'] >= 3.0) & (g['rating'] < 4.0) & (g['discount_percent'].between(0.2, 0.6))],
        "saverpro": lambda g: g[g['rating'] >= 4.0].sort_values(by='discount_percent', ascending=False),
        "savermed": lambda g: g[(g['rating'] >= 3.0) & (g['rating'] < 4.0)].sort_values(by='discount_percent', ascending=False),
        "saverfree": lambda g: g[g['discount_percent'] >= np.percentile(g['discount_percent'], 75)],
        "socialproof": lambda g: g[g['popularity_score'] >= np.percentile(g['popularity_score'], 75)]
    }

    for persona, logic in personas.items():
        df_persona = df.copy()
        df_persona['purchased'] = 0  # Default: no purchase

        for category, group in df.groupby('main_category'):
            filtered_group = logic(group)
            if not filtered_group.empty:
                # Set purchased = 1 for all matching products in this category
                condition = df_persona['name'].isin(filtered_group['name']) & (df_persona['main_category'] == category)
                df_persona.loc[condition, 'purchased'] = 1

        # Keep only the required columns
        df_persona = df_persona[["name", "main_category", "sub_category", "rating", "no_of_ratings", "discount_price", "actual_price", "purchased"]]

        # Save the persona file
        output_path = os.path.join(output_folder, f"products_{persona}.csv")
        df_persona.to_csv(output_path, index=False)
        print(f"✅ Generated: {output_path}")

    print(f"\n✅✅ All persona datasets generated at: {output_folder}")

# Example usage
# persona_datasets('/content/master_products_cleaned.csv', '/content/synthetic_personas')
