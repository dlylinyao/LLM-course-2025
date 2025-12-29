import pandas as pd
import dspy
import re

try:
    
    lm = dspy.LM(model="ollama_chat/llama3", api_base="http://localhost:11434", timeout=60000)
except Exception:
    try:
        lm = dspy.Ollama(model="llama3", timeout=60000)
    except AttributeError:
        print("Error: Could not initialize Ollama. Ensure 'dspy' is updated and Ollama is running.")
        exit(1)

dspy.configure(lm=lm)

# Define the DSPy Signature
class GenerateMisspellings(dspy.Signature):
    """
    You are a search engine quality tester.
    Task: Generate N distinct misspelling variants for the given search Query.
    
    Guidelines:
    1.  **PROTECT ABBREVIATIONS**: Do NOT alter words that are all uppercase (e.g., JFK, NYC, USA, IBM).
    2.  **VARIETY**: Ensure the variants cover these error types:
        - Phonetic (e.g., "mashine" for "machine")
        - Omission (e.g., "learnin" for "learning")
        - Transposition (e.g., "framewrok" for "framework")
        - Repetition (e.g., "apppplications" for "applications")
    3.  **OUTPUT**: Return ONLY the list of misspelled query strings.
    """
    
    query = dspy.InputField(desc="The original search query.")
    protected_terms = dspy.InputField(desc="List of abbreviations/terms to keep exact.")
    n_variants = dspy.InputField(desc="Number of variants to generate.")
    misspelled_queries = dspy.OutputField(desc="List of N misspelled queries.", format=list)

# Helper to find abbreviations 
def find_abbreviations(text):

    return re.findall(r'\b[A-Z]{2,}\b', text)

# Main Execution Logic
def main():
    try:
        df = pd.read_csv('web_search_queries.csv')
        print(f"Loaded {len(df)} queries.")
    except FileNotFoundError:
        print("Error: web_search_queries.csv not found.")
        return

    generator = dspy.Predict(GenerateMisspellings)
    
    results = []

    print("Generating misspellings... (this may take a moment)")
    for index, row in df.iterrows():
        original_query = row['Query']
        topic = row['Topic']
        
        abbrevs = find_abbreviations(original_query)
        protected_str = ", ".join(abbrevs) if abbrevs else "None"

        try:
            response = generator(
                query=original_query, 
                protected_terms=protected_str, 
                n_variants=5
            )
            
            raw_output = response.misspelled_queries
            if isinstance(raw_output, str):
                variants = [line.strip() for line in raw_output.split('\n') if line.strip()]
                variants = [re.sub(r'^\d+\.\s*', '', v) for v in variants]
            else:
                variants = raw_output

            print(f"\nOriginal: {original_query}")
            print(f"Variants: {variants}")
            
            for v in variants:
                results.append({
                    'Topic': topic,
                    'Original_Query': original_query,
                    'Misspelled_Query': v
                })
                
        except Exception as e:
            print(f"Error processing '{original_query}': {e}")

    output_df = pd.DataFrame(results)
    output_df.to_csv('synthetic_queries_output.csv', index=False)
    print("\nDone! Results saved to 'synthetic_queries_output.csv'")

if __name__ == "__main__":
    main()

# you can use DSPY (https://github.com/stanfordnlp/dspy), but you can also choose another method of interacting with an LLM
#dspy.settings.configure(lm=llm)

# Task: implement a method, that will take a query string as input and produce N misspelling variants of the query.
# These variants with typos will be used to test a search engine quality.
# Example
# Query: machine learning applications
# Possible Misspellings:
# "machin learning applications" (missing "e" in "machine")
# "mashine learning applications" (phonetically similar spelling of "machine")
# "machine lerning aplications" (missing "a" in "learning" and "p" in "applications")
# "machin lerning aplications" (combining multiple typos)
# "mahcine learing aplication" (transposed letters in "machine" and typos in "learning" and "applications")
#
# Questions:
# 1. Does the search engine produce the same results for all the variants?
# 2. Do all variants make sense?
# 3. How to improve robustness of the method, for example, skip known abbreviations, like JFK or NBC.
# 4. Can you test multiple LLMs and figure out which one is the best?
# 5. Do the misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?