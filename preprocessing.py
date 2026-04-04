'''Preprocessing script for our project. 
It is 28 gigabytes of data, I only process 10% of it here
but even then it destroyed my laptop. You can run it for a while.
I made it so it reads in chunks and saves it on the disk incrementally,
you can check its input after few chunks have been processed
and see if it is good enough
I applied stemming too even though lemmatization might be better,
but the assignment needed us to do stemming. Download all 28gbs of data
and put it in a folder called 'data' and run the script.
You need at least 16gbs for this to not crash, it takes up around 6gbs
and then empties the RAM after the chunk is processed.'''

import pandas as pd
import csv
import spacy
import spacy.cli
from nltk.stem import PorterStemmer
from collections import Counter
import os
import time

class FakeNewsPreprocessor:
    # Download spaCy model and initialize stop words and stemmer.
    def __init__(self):
        print("Loading spaCy English model...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.stop_words = self.nlp.Defaults.stop_words
        self.stemmer = PorterStemmer()

        # The columns we want to keep, I removed type from stemming and kept the full word because we will be using it for  'labels' later on.
        self.text_columns = [
            "content", "title", "authors", "keywords", "source"
        ]
        # Added this to keep track of the label column without changing it.
        self.label_column = "type"

        # Vocabulary counters for analysis
        self.vocab_before = Counter()
        self.vocab_after_stopwords = Counter()
        self.vocab_after_stemming = Counter()

    def clean_text_series(self, series):
        """Fill NaN, turn to lowercase, remove URLs and HTML."""
        s = series.fillna("").str.lower()
        #s = s.str.replace(r"http\S+|www\S+", "", regex=True)
        #s = s.str.replace(r"<.*?>", "", regex=True)
        return s

    # Tokenize one column at a time using spaCy, yielding lists of tokens. This is used in process_chunk, and separated to not blow up the RAM.
    def tokenize_column(self, texts):
        
        for doc in self.nlp.pipe(texts, batch_size=500):
            yield [token.text for token in doc if token.is_alpha]

    def process_chunk(self, chunk):
        """Clean, tokenize, remove stopwords, stem, update vocab, return processed DataFrame."""
        
        # Clean text feature columns
        for col in self.text_columns:
            if col in chunk.columns:
                chunk[col] = self.clean_text_series(chunk[col])

        # Filter invalid or empty content
        if "content" in chunk.columns:
            chunk = chunk[chunk["content"].str.strip() != ""]
            chunk = chunk.drop_duplicates(subset=["content", "title"])

        # Tokenize, remove stopwords, stem, and update vocab for the columns in text_columns
        for col in self.text_columns:
            if col in chunk.columns:
                processed_texts = []

                # Tokenize using spaCys' pipe. 
                for doc in self.nlp.pipe(chunk[col], batch_size=500):
                    tokens = [token.text for token in doc if token.is_alpha]
                    
                    # Update vocab before stopwords removal
                    self.vocab_before.update(tokens)

                    # Remove stopwords and update vocab
                    tokens = [w for w in tokens if w not in self.stop_words]
                    self.vocab_after_stopwords.update(tokens)

                    # Apply stemming and update vocab
                    tokens = [self.stemmer.stem(w) for w in tokens]
                    self.vocab_after_stemming.update(tokens)

                    processed_texts.append(" ".join(tokens))

                chunk[col] = processed_texts

        return chunk

    # Change chunksize if your RAM is struggling, but it will take longer to process. Sample fraction is how much of the data you like to process, default is 10%.
    def load_and_process(self, input_csv, output_csv, chunksize=30_000, sample_frac=0.1):
        """Process CSV in memory-efficient batches, save to disk incrementally, compute vocab stats."""
        # The pandas CSV reader will crash if this is not here. Use high numbers.
        csv.field_size_limit(10_000_000)

        if os.path.exists(output_csv):
            os.remove(output_csv)

        # Save the first row so we could see how the data looks like.
        first_rows_path = "data/chunk_first_rows.csv"
        if os.path.exists(first_rows_path):
            os.remove(first_rows_path)

        start_time = time.time()
        batch_num = 0

        # Dynamically detect available columns
        all_cols = pd.read_csv(input_csv, nrows=1).columns # Every column
        used_cols = [col for col in self.text_columns + [self.label_column] if col in all_cols]

        for chunk in pd.read_csv(
            input_csv,
            usecols=used_cols,
            chunksize=chunksize,
            engine="python",
            on_bad_lines="skip"
        ):
            batch_num += 1

            # Save first row of each chunk
            first_row = chunk.head(1)
            first_row.to_csv(
                first_rows_path,
                mode="a",
                index=False,
                header=not os.path.exists(first_rows_path),
            )

            print(f"\nProcessing chunk {batch_num}...")

            # Sample a fraction of the chunk to reduce memory usage, then process and save it incrementally.
            chunk = chunk.sample(frac=sample_frac, random_state=42)
            chunk = self.process_chunk(chunk)
            
            # Append processed chunk to output CSV
            write_header = not os.path.exists(output_csv)
            chunk.to_csv(output_csv, mode="a", index=False, header=write_header)

            print(f"Chunk {batch_num} processed and saved.")

        elapsed = time.time() - start_time
        print(f"\nAll batches processed in {elapsed:.2f} seconds.")
        print(f"Processed data saved to: {output_csv}")

        self.report_vocab_statistics()


    def report_vocab_statistics(self):
        """Print vocabulary sizes and reduction rates."""
        vocab_before_size = len(self.vocab_before)
        vocab_after_stop_size = len(self.vocab_after_stopwords)
        vocab_after_stem_size = len(self.vocab_after_stemming)

        # Ensure no division by zero.
        if vocab_before_size > 0:
            stopword_reduction = 1 - vocab_after_stop_size / vocab_before_size
        else:
            stopword_reduction = 0

        if vocab_after_stop_size > 0:
            stemming_reduction = 1 - vocab_after_stem_size / vocab_after_stop_size
        else:
            stemming_reduction = 0

        print("\n Vocabulary Statistics:")
        print(f"Vocabulary size before stopwords: {vocab_before_size}")
        print(f"Vocabulary size after stopwords removal: {vocab_after_stop_size}")
        print(f"Reduction rate after stopwords removal: {stopword_reduction:.2%}")
        print(f"Vocabulary size after stemming: {vocab_after_stem_size}")
        print(f"Reduction rate after stemming: {stemming_reduction:.2%}")

# Run the preprocessing
processor = FakeNewsPreprocessor()
input_path = "data/news_cleaned_2018_02_13.csv"
output_path = "data/news_cleaned_2018_02_13_cleaned_10pct.csv"


# You can change the parameters if you wanted to here besides the defaults.
processor.load_and_process(
    input_csv=input_path,
    output_csv=output_path
)