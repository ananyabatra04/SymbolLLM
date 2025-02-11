from pyswip import Prolog
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")

def fetch_prolog_code(prompt):
    """
    Uses Symbol LLM model to generate the Prolog code based on the given prompt.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False, padding=True)

        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1
        )

        prolog_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated Prolog Code:\n", prolog_code)
        return prolog_code.strip()

    except Exception as e:
        print(f"Error fetching Prolog code from Hugging Face: {e}")
        return None

def format_statements(prolog_code):
    """
    Reformat Prolog code to ensure statements are properly grouped and complete.
    """
    statements, temp = [], []

    for line in prolog_code.splitlines(): # split by new line
        line = line.strip()
        if line and not line.startswith("%"): # ignore comments and empty lines
            temp.append(line)
            if line.endswith('.'):
                statements.append(" ".join(temp))
                temp.clear()

    return statements

def execute_code(prolog_code, query):
    """
    Load and execute Prolog code to run a query against it.
    """
    prolog = Prolog()
    statements = format_statements(prolog_code)

    try:
        for smt in statements:
            prolog.assertz(smt.rstrip('.'))  # Add fact or rule to knowledge base

        return list(prolog.query(query))  # Execute the query
    except Exception as error:
        raise Exception(f"Prolog execution failed: {error}")

def run_baseline_test():
    prompt = '''
      Generate Prolog code that defines parent-child relationships for Mary and others, and includes a rule for grandparent. Define the following parent-child relationships as facts and define a rule for grandparent: A grandparent is someone who is the parent of a parent.
      '''

    prolog_code = fetch_prolog_code(prompt)

    if prolog_code:

        test_query = "grandparent(mary, X)."  # defining the query to run

        try:
            results = execute_code(prolog_code, test_query)
            print("Query Results:", results)
        except Exception as error:
            print(f"Error during Prolog execution: {error}")
    else:
        print("Failed to fetch Prolog code.")

if __name__ == "__main__":
    run_baseline_test()
