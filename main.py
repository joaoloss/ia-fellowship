from utils.pdf2mat import PDF2Matrix
from utils.type_resolution import TypeResolver
from utils.heuristic import Heuristic

from openai import OpenAI
from time import time
from pydantic import create_model
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv
import yaml
import json
import base64
from argparse import ArgumentParser
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_DIR = Path("files")
INPUT_JSON = "dataset.json"
PRICE_PER_1M_INPUT_TOKENS = 0.25 # USD (02/11/2025)
PRICE_PER_1M_OUTPUT_TOKENS = 2.0 # USD (02/11/2025)

PROMPT_TEMPLATE_1 = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no texto representado a estrutura de um PDF.

# Instruções
* Retorne apenas o JSON solicitado, sem explicações adicionais.
* Inclua todos os campos do YAML, preenchendo com null quando a informação não estiver disponível (não use strings vazias como valor de algum campo).
* Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no texto do PDF. Se não estiver explicitamente presente ou reconhecível, defina o valor como null. Não substitua ou use termos similares para preencher a chave.

# YAML de requisição
{request_yaml}

# Texto representando a estrutura do PDF
{pdf_text}

Comece.
"""

PROMPT_TEMPLATE_2 = """

# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no PDF.

# Instruções
* Retorne apenas o JSON solicitado, sem explicações adicionais.
* Inclua todos os campos do YAML, preenchendo com null quando a informação não estiver disponível (não use strings vazias como valor de algum campo).
* Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no PDF. Se não estiver explicitamente presente ou reconhecível, defina o valor como null. Não substitua ou use termos similares para preencher a chave.

# YAML de requisição
{request_yaml}

Comece.
"""

client = OpenAI(api_key=OPENAI_API_KEY)
type_resolver = TypeResolver()
heuristic_cache = Heuristic()

def parse_arguments():
    parser = ArgumentParser(description="PDF Extraction with LLM")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1,
                        help="Version of the extraction method to use (1 or 2). Default is 1.")
    parser.add_argument("--use-heuristic", action="store_true",
                        help="Enable heuristic caching mechanism. Default is False.")
    return parser.parse_args()

def llm_response_v1(input_schema: dict, matrix: list):
    """
    Envia a representação em texto do PDF (matriz) junto com o schema de extração.

    Essa versão tende a ser mais rápida e consumir menos tokens, mas pode ser menos precisa dependendo do layout do PDF.
    """

    mat_to_str = [f"Row {i+1}: " + " | ".join(row) for i, row in enumerate(matrix)]
    mat_to_str = "\n".join(mat_to_str)
    
    output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
    OutputModelStructure = create_model("OutputModelStructure", **output_structure)

    yaml_schema = yaml.dump(input_schema, allow_unicode=True)

    prompt = PROMPT_TEMPLATE_1.format(request_yaml=yaml_schema, pdf_text=mat_to_str).strip()

    response = client.responses.parse(model="gpt-5-mini-2025-08-07",
                                      text_format=OutputModelStructure,
                                      reasoning={"effort":"minimal"},
                                      text={"verbosity":"low"},
                                      input=prompt)
    return response

def llm_response_v2(input_schema: dict, pdf_path: str):
    """
    Envia o PDF diretamente para o modelo (usando base64) junto com o schema de extração.

    Essa versão tende a ser mais lenta e consumir mais tokens, mas pode ser mais precisa dependendo do layout do PDF.
    """

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
    OutputModelStructure = create_model("OutputModelStructure", **output_structure)

    yaml_schema = yaml.dump(input_schema, allow_unicode=True)

    prompt = PROMPT_TEMPLATE_2.format(request_yaml=yaml_schema).strip()

    response = client.responses.parse(model="gpt-5-mini-2025-08-07",
                                      text_format=OutputModelStructure,
                                      reasoning={"effort":"minimal"},
                                      input=[
                                          {
                                              "role": "user",
                                              "content": [
                                                    {"type": "input_text", "text": prompt},
                                                    {"type": "input_file", "filename": str(pdf_path), "file_data": f"data:application/pdf;base64,{pdf_base64}"}
                                              ]
                                          }
                                      ])
    return response

def inference_cost_estimation(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    total_cost = input_cost + output_cost
    return total_cost

def main():
    args = parse_arguments()
    version = args.version
    use_heuristic = args.use_heuristic

    with open(INPUT_JSON) as f:
        input_json = json.load(f)
    
    input_files = os.listdir(INPUT_DIR)
    result_json = list()
    for item in input_json:
        start_time = time()
        pdf_file_name = item["pdf_path"]

        if pdf_file_name not in input_files:
            print(f"File {pdf_file_name} not found in {INPUT_DIR}. Skipping...")
            continue

        pdf_path = INPUT_DIR / pdf_file_name        
        print(f"Processing file: {pdf_file_name}")

        pdf2matrix = PDF2Matrix(pdf_path)
        matrix = pdf2matrix.get_matrix()

        request_schema = dict(item["extraction_schema"])
        print(list(request_schema.keys()))

        result = dict()
        if use_heuristic:
            heuristic_cache.heuristic_preprocessing(label=item["label"],
                                                    request_schema=request_schema,
                                                    partial_result=result,
                                                    pdf_matrix=matrix)
            print(list(request_schema.keys()))

        if version == 1:
            response = llm_response_v1(input_schema=request_schema, matrix=matrix)
        else:
            response = llm_response_v2(input_schema=request_schema, pdf_path=pdf_path)

        llm_formatted_output = dict(response.output_parsed)
        result.update(llm_formatted_output)
        
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.\n\n")

        # print(response.usage)
        # return

        result_and_metadata = {
            "pdf_path": pdf_file_name,
            "result": result,
            "latency_seconds": round(elapsed_time, 2),
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "estimated_cost_usd": f"{inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}"
        }
        result_json.append(result_and_metadata)

        if use_heuristic:
            heuristic_cache.heuristic_update(partial_result=llm_formatted_output,
                                             label=item["label"],
                                             pdf_matrix=pdf2matrix)
    with open(f"output_cache_results_v{version}.json", "w") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    
    with open(f"heuristic_cache_v{version}.json", "w") as f:
        json.dump(heuristic_cache.get_cache(), f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()