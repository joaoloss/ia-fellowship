from utils.pdf2mat import PDF2Matrix
from utils.type_resolution import TypeResolver

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
client = OpenAI(api_key=OPENAI_API_KEY)
heuristic_cache = dict()
type_resolver = TypeResolver()

PRICE_PER_1M_INPUT_TOKENS = 0.25 # USD (02/11/2025)
PRICE_PER_1M_OUTPUT_TOKENS = 2.0 # USD (02/11/2025)
HEURISTIC_CACHE_THRESHOLD = 0.6

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
    
    with open(INPUT_DIR / pdf_path, "rb") as f:
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
                                                    {"type": "input_file", "filename": pdf_path, "file_data": f"data:application/pdf;base64,{pdf_base64}"}
                                              ]
                                          }
                                      ])
    return response

def inference_cost_estimation(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    total_cost = input_cost + output_cost
    return total_cost

def heuristic_update(result:dict, label:str, pdf_matrix:PDF2Matrix):
    if label not in heuristic_cache:
        heuristic_cache[label] = dict()
    
    for key, value in result.items():
        if not value:
            continue
        if key not in heuristic_cache[label]:
            heuristic_cache[label][key] = {
                "count": 0,
                "heuristics": [{"type": type_resolver.resolve(value),
                                "position": pdf_matrix.get_position_of_text(value),
                                "match_count": 1}]
            }            
        else:
            for record in heuristic_cache[label][key]["heuristics"]:
                if record["type"] == type_resolver.resolve(value) and record["position"] == pdf_matrix.get_position_of_text(value):
                    record["match_count"] += 1
                    break
            else:
                heuristic_cache[label][key]["heuristics"].append({"type": type_resolver.resolve(value),
                                                                  "position": pdf_matrix.get_position_of_text(value),
                                                                  "match_count": 1})
                
                # Keep only top 3 heuristics per key based on match_count
                heuristic_cache[label][key]["heuristics"] = sorted(heuristic_cache[label][key]["heuristics"], key=lambda x: x["match_count"], reverse=True)[:3]
        heuristic_cache[label][key]["count"] += 1

def heuristic_preprocessing(label:str, request_schema:dict, result:dict, pdf_matrix:list[list[str]]):
    if label not in heuristic_cache:
        return
    request_schema_keys = list(request_schema.keys())
    for key in request_schema_keys:
        if key not in heuristic_cache[label]:
            continue

        for record in heuristic_cache[label][key]["heuristics"]:
            # Try to find the most reliable heuristic for this key
            if record["match_count"] / heuristic_cache[label][key]["count"] > HEURISTIC_CACHE_THRESHOLD:
                position = record["position"]
                if position is None:
                    break
                
                row_index, col_index = position
                if col_index == -1: # fuzzy row match or 1D row
                    text = " ".join(pdf_matrix[row_index])
                    text = re.sub(r"\s+", " ", text).strip()
                    result[key] = text
                else:
                    try:
                        if len(pdf_matrix[row_index]) > 1: # 2D row
                            result[key] = pdf_matrix[row_index][col_index]
                        else: # 1D row
                            result[key] = pdf_matrix[row_index]
                    except IndexError: # Not found field
                        result[key] = None
                
                request_schema.pop(key)
                break

def main():
    args = parse_arguments()
    version = args.version

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
        
        print(f"Processing file: {pdf_file_name}")

        pdf2matrix = PDF2Matrix(str(INPUT_DIR / pdf_file_name))
        matrix = pdf2matrix.get_matrix()

        request_schema = dict(item["extraction_schema"])

        print("Initial request schema keys:", list(request_schema.keys()))

        result_item = dict()
        if args.use_heuristic:
            heuristic_preprocessing(label=item["label"],
                                    request_schema=request_schema,
                                    result=result_item,
                                    pdf_matrix=matrix)
            print("Request schema keys after heuristic preprocessing:", list(request_schema.keys()))

        if version == 1:
            response = llm_response_v1(request_schema, matrix)
        else:
            response = llm_response_v2(request_schema, pdf_file_name)

        llm_formatted_output = dict(response.output_parsed)
        result_item.update(llm_formatted_output)

        if args.use_heuristic:
            heuristic_update(result=llm_formatted_output,
                            label=item["label"],
                            pdf_matrix=pdf2matrix)
        
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.")

        # print(response.usage)
        # return

        result_item = {
            "pdf_path": pdf_file_name,
            "result": result_item,
            "latency_seconds": round(elapsed_time, 2),
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "estimated_cost_usd": f"{inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}"
        }
        result_json.append(result_item)

    with open(f"output_cache_results_v{version}.json", "w") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    
    with open(f"heuristic_cache_v{version}.json", "w") as f:
        json.dump(heuristic_cache, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()