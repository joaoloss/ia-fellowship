from utils.pdf2mat import PDF2Matrix

from openai import OpenAI
from time import time
from pydantic import BaseModel, create_model
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv
import yaml
import json
import base64
from argparse import ArgumentParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_DIR = Path("files")
INPUT_JSON = "dataset.json"
client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_TEMPLATE_1 = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no texto representado a estrutura de um PDF.

# YAML de requisição
{request_yaml}

# Texto representando a estrutura do PDF
{pdf_text}

# Instruções
* Retorne apenas o JSON solicitado, sem explicações adicionais.
* Inclua todos os campos do YAML, preenchendo com null quando a informação não estiver disponível (não use strings vazias como valor de algum campo).
* Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no texto do PDF. Se não estiver explicitamente presente ou reconhecível, defina o valor como null. Não substitua ou use termos similares para preencher a chave.
"""

PROMPT_TEMPLATE_2 = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no PDF.

# YAML de requisição
{request_yaml}

# Instruções
* Retorne apenas o JSON solicitado, sem explicações adicionais.
* Inclua todos os campos do YAML, preenchendo com null quando a informação não estiver disponível (não use strings vazias como valor de algum campo).
* Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no PDF. Se não estiver explicitamente presente ou reconhecível, defina o valor como null. Não substitua ou use termos similares para preencher a chave.
"""

def llm_response_v1(input_schema: dict, pdf_path: str):
    """
    Extrai o texto do PDF e envia para o modelo junto com o schema de extração.
    O texto do PDF é passado para o modelo em formato de matriz para melhor compreensão da estrutura.

    Essa versão tende a ser mais rápida e consumir menos tokens, mas pode ser menos precisa dependendo do layout do PDF.
    """
    
    pdf2matrix = PDF2Matrix(str(INPUT_DIR / pdf_path))
    matrix = pdf2matrix.get_matrix()

    mat_to_str = [f"Row {i+1}: " + " | ".join(row) for i, row in enumerate(matrix)]
    mat_to_str = "\n".join(mat_to_str)

    output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
    OutputModelStructure = create_model("OutputModelStructure", **output_structure)

    yaml_schema = yaml.dump(input_schema, allow_unicode=True)

    prompt = PROMPT_TEMPLATE_1.format(request_yaml=yaml_schema, pdf_text=mat_to_str).strip()

    response = client.responses.parse(model="gpt-5-mini-2025-08-07",
                                      text_format=OutputModelStructure,
                                      reasoning={"effort":"minimal"},
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

def parse_arguments():
    parser = ArgumentParser(description="PDF Extraction with LLM")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1,
                        help="Version of the extraction method to use (1 or 2). Default is 1.")
    return parser.parse_args()

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

        if version == 1:
            response = llm_response_v1(item["extraction_schema"], pdf_file_name)
        else:
            response = llm_response_v2(item["extraction_schema"], pdf_file_name)

        end_time = time()
        elapsed_time = end_time - start_time
        print(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.")

        result_item = {
            "pdf_path": pdf_file_name,
            "result": dict(response.output_parsed),
            "latency_seconds": round(elapsed_time, 2),
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
        result_json.append(result_item)

    with open(f"output_results_v{version}.json", "w") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()