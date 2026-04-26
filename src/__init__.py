import json
import pandas as pd
import re


def load_and_clean_arxiv(file_path, target_records=50000):
    data = []
    print(f"Scanning for {target_records} modern CS papers (2018+)...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(data) >= target_records:
                break

            doc = json.loads(line)
            categories = doc.get('categories', '')

            # Fast string check before parsing dates
            if 'cs.' not in categories:
                continue

            update_date = doc.get('update_date', '')
            year = int(update_date[:4])  # Faster than splitting

            if year >= 2018:
                data.append({
                    'id': doc.get('id'),
                    'categories': categories,
                    'abstract': doc.get('abstract'),
                    'year': year
                })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records. Cleaning text...")

    # ---------------------------------------------------------
    # THE ENGINEERING FIX: Bypass Pandas .str overhead
    # List comprehensions compile closer to C and are much faster
    # ---------------------------------------------------------

    # Pre-compile the regex engine for speed
    whitespace_re = re.compile(r'\s+')

    # Clean abstracts using pure python
    raw_abstracts = df['abstract'].tolist()
    cleaned_abstracts = [whitespace_re.sub(' ', text.replace('\n', ' ')).strip() for text in raw_abstracts]
    df['abstract'] = cleaned_abstracts

    print("Extracting primary category...")
    # Extract the first cs. category
    df['primary_category'] = [
        next((cat for cat in cats.split() if cat.startswith('cs.')), None)
        for cats in df['categories']
    ]

    df = df.dropna(subset=['primary_category', 'abstract'])

    return df


# Start with 50,000. Prove your pipeline works before jumping to 1M.
# noinspection PyInterpreter
df_clean = load_and_clean_arxiv('data/raw/arxiv-metadata-oai-snapshot.json', target_records=100000)

print(df_clean.head())
print("\nDataset Shape:", df_clean.shape)