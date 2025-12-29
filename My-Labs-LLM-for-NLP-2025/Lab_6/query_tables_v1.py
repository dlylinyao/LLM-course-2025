from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Source: https://medium.com/@jitsins/query-complex-pdfs-in-natural-language-with-llmsherpa-ollama-llama3-8b-13b4782243de
# To install:
# 1. run https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
# 2. install and run ollama:
# ollama pull llama3
# ollama run llama3
# 3. Install docker and run:
# docker pull ghcr.io/nlmatics/nlm-ingestor:latest
# docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
# This will expose the api link “http://localhost:5010/api/parseDocument?renderFormat=all” for you to utilize in your code.

# Initialize LLm
llm = Ollama(model="llama3", request_timeout=120.0)

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# Read PDF
doc = pdf_reader.read_pdf(pdf_url)

# Get data from all Section
all_context = ""
for section in doc.sections():
    all_context += section.to_html(include_children=True, recurse=True) + "\n"

# test 1:  Baseline
question_1 = "What were the Google Cloud revenues in Q1 2023 and Q1 2024?"
print(f"\n--- Question 1: {question_1} ---")
resp = llm.complete(f"Read the provided tables and answer: {question_1}. Context:\n{all_context}")
print(resp.text)

# test 2:
question_2 = "Calculate the exact dollar amount increase in 'Google Services' operating income from 2023 to 2024 based on the table."
print(f"\n--- Question 2: {question_2} ---")
resp = llm.complete(f"Read the provided tables, find the operating income for Google Services for both years, and calculate the difference. Show your math. Context:\n{all_context}")
print(resp.text)

# test3: 
question_3 = "Sum up the 'TAC' (Traffic Acquisition Costs) paid to distribution partners and the TAC paid to Google Network partners for Q1 2024."
print(f"\n--- Question 3: {question_3} ---")
resp = llm.complete(f"Read the provided tables and perform the calculation: {question_3}. Context:\n{all_context}")
print(resp.text)