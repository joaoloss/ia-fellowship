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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging
import sys

# Logger setup
logger = logging.getLogger("my_logger")
logger.propagate = False
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Constants and configurations
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_DIR = Path("files")
INPUT_JSON = "dataset.json"
PRICE_PER_1M_INPUT_TOKENS = 0.25 # USD (02/11/2025)
PRICE_PER_1M_OUTPUT_TOKENS = 2.0 # USD (02/11/2025)

PROMPT_TEMPLATE_1 = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no conteúdo do PDF. O conteúdo do PDF é fornecido em duas representações: uma estruturada (matriz, mantendo a formatação) e uma contínua (texto bruto).
Use ambas as representações para extrair as informações solicitadas da forma mais precisa e completa possível, perda de informação é um erro crítico.

# Instruções
* Retorne apenas o JSON solicitado, sem explicações adicionais.
* Inclua todos os campos do YAML, preenchendo com null quando a informação não estiver disponível (não use strings vazias como valor de algum campo).
* Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no texto do PDF. Se não estiver explicitamente presente ou reconhecível, defina o valor como null. Não substitua ou use termos similares para preencher a chave.

# YAML de requisição
{request_yaml}

# Texto representando a estrutura do PDF
{pdf_text}

# Representação em um texto contínuo do mesmo PDF
{raw_text}

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
    parser.add_argument("--no-heuristic", action="store_true",
                        help="Disable heuristic caching mechanism. Default is False.")
    return parser.parse_args()

def llm_response_v2(input_schema: dict, matrix: list):
    """
    Envia a representação em texto do PDF (matriz) junto com o schema de extração.

    Essa versão tende a ser mais rápida e consumir menos tokens, mas pode ser menos precisa dependendo do layout do PDF.
    """

    mat_to_str = [f"Row {i+1}: " + " | ".join(row) for i, row in enumerate(matrix)]
    mat_to_str = "\n".join(mat_to_str)

    raw_text = " ".join([" ".join(row) for row in matrix])

    with open("debug_outputs/raw_text.txt", "a", encoding="utf-8") as f:
        f.write(raw_text)
        f.write("\n\n")

    os.makedirs("debug_outputs", exist_ok=True)
    with open(f"debug_outputs/pdf_representation.txt", "a", encoding="utf-8") as f:
        f.write(raw_text + "\n\n" + mat_to_str + "\n\n" + ("="*80) + "\n\n")
    
    output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
    OutputModelStructure = create_model("OutputModelStructure", **output_structure)

    yaml_schema = yaml.dump(input_schema, allow_unicode=True)

    prompt = PROMPT_TEMPLATE_1.format(request_yaml=yaml_schema,
                                      pdf_text=mat_to_str,
                                      raw_text=raw_text).strip()

    response = client.responses.parse(model="gpt-5-mini-2025-08-07",
                                      text_format=OutputModelStructure,
                                      reasoning={"effort":"minimal"},
                                      input=prompt)
    return response

def llm_response_v1(input_schema: dict, pdf_path: str):
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
        
def main_w_threads():
    """
    Main function to process PDFs using multithreading.
    Although this improves performance, it introduces two issues:
    (1) The processing order is no longer guaranteed to match the input order.
    (2) The heuristic cache becomes less effective, since concurrent workers may use it
        before it has been updated by previous outputs.
    """

    GLOBAL_LOCK = Lock()
    HEURISTIC_LOCK = Lock()
    
    def worker(pdf_path: str,
                item: dict,
                version: int,
                use_heuristic: bool,
                result_json: list):
        start_time = time()
        try:
            pdf2matrix = PDF2Matrix(pdf_path)
            matrix = pdf2matrix.create_matrix_representation()
        except Exception as e:
            logger.error(f"Error generating PDF matrix for {pdf_path}: {e}")
            use_heuristic = True # Disable heuristic for this iteration
            version = 1 # Fallback to version 1 if matrix generation fails

        request_schema = dict(item["extraction_schema"])

        result = dict()
        if use_heuristic:
            with HEURISTIC_LOCK:
                result = heuristic_cache.heuristic_preprocessing(label=item["label"],
                                                                request_schema=request_schema,
                                                                pdf_matrix_representation=matrix)
            heuristic_hits = list(result.keys())
            logger.info(f"Heuristic hits for {pdf_path}: {heuristic_hits}")

            # Remove already filled keys from the request schema
            request_schema = {k: v for k, v in request_schema.items() if k not in heuristic_hits}

        if version == 1:
            response = llm_response_v1(input_schema=request_schema, pdf_path=pdf_path)
        else:
            response = llm_response_v2(input_schema=request_schema, matrix=matrix)

        llm_formatted_output = dict(response.output_parsed)
        result.update(llm_formatted_output)
        if use_heuristic:
            with HEURISTIC_LOCK:
                heuristic_cache.heuristic_update(result=llm_formatted_output,
                                                label=item["label"],
                                                pdf_matrix=pdf2matrix)
        
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Processed {pdf_path} in {elapsed_time:.2f} seconds.\n\n")

        result_and_metadata = {
            "pdf_path": pdf_path,
            "extraction_schema": result,
            "latency_seconds": round(elapsed_time, 2),
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "estimated_cost_usd": f"{inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}",
        }

        if use_heuristic:
            result_and_metadata["heuristic_hits"] = heuristic_hits

        with GLOBAL_LOCK:
            result_json.append(result_and_metadata)
            with open(OUTPUT_JSON_FILE, "w") as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
    
    args = parse_arguments()
    version = args.version
    use_heuristic = not args.no_heuristic

    logger.debug(f"Using version: {version}")
    logger.debug(f"Heuristic caching enabled: {use_heuristic}\n")

    with open(INPUT_JSON) as f:
        input_json = json.load(f)
    
    input_files = os.listdir(INPUT_DIR)
    result_json = list()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for item in input_json:
            
            pdf_file_name = item["pdf_path"]

            if pdf_file_name not in input_files:
                logger.warning(f"File {pdf_file_name} not found in {INPUT_DIR}. Skipping...")
                continue

            pdf_path = INPUT_DIR / pdf_file_name
            logger.info(f"Processing file: {pdf_file_name}")

            executor.submit(worker,
                            pdf_path=str(pdf_path),
                            item=item,
                            version=version,
                            use_heuristic=use_heuristic,
                            result_json=result_json)

def main():
    args = parse_arguments()
    version = args.version
    use_heuristic = not args.no_heuristic
    disable_heuristic = False # To handle errors in PDF matrix generation
    use_version = version
    output_json_file = f"output_results_v{version}.json"

    logger.debug(f"Using version: {version}")
    logger.debug(f"Heuristic caching enabled: {use_heuristic}\n")

    with open(INPUT_JSON) as f:
        input_json = json.load(f)
    
    with open(output_json_file, "w") as f:
        f.write("[]")  # Initialize output file
    
    input_files = os.listdir(INPUT_DIR)
    result_json = list()
    for item in input_json:
        start_time = time()
        pdf_file_name = item["pdf_path"]

        if pdf_file_name not in input_files:
            logger.warning(f"File {pdf_file_name} not found in {INPUT_DIR}. Skipping...")
            continue

        pdf_path = INPUT_DIR / pdf_file_name
        logger.info(f"Processing file: {pdf_file_name}")

        try:
            pdf2matrix = PDF2Matrix(pdf_path)
            matrix = pdf2matrix.create_matrix_representation()
        except Exception as e:
            logger.error(f"Error generating PDF matrix for {pdf_file_name}: {e}")
            disable_heuristic = True # Disable heuristic for this iteration
            use_version = 2 # Fallback to version 2 if matrix generation fails

        request_schema = dict(item["extraction_schema"])

        result = dict()
        if use_heuristic and not disable_heuristic:
            result = heuristic_cache.heuristic_preprocessing(label=item["label"],
                                                            request_schema=request_schema,
                                                            pdf_matrix_representation=matrix)
            heuristic_hits = list(result.keys())
            logger.info(f"Heuristic hits for {pdf_file_name}: {heuristic_hits}")

            # Remove already filled keys from the request schema
            request_schema = {k: v for k, v in request_schema.items() if k not in heuristic_hits}

        if use_version == 1:
            response = llm_response_v1(input_schema=request_schema, pdf_path=pdf_path)
        else:
            response = llm_response_v2(input_schema=request_schema, matrix=matrix)

        llm_formatted_output = dict(response.output_parsed)
        result.update(llm_formatted_output)
        
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.\n\n")

        result_and_metadata = {
            "pdf_path": pdf_file_name,
            "extraction_schema": result,
            "latency_seconds": round(elapsed_time, 2),
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "estimated_cost_usd": f"{inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}"
        }
        if use_heuristic and not disable_heuristic:
            result_and_metadata["heuristic_hits"] = heuristic_hits
        result_json.append(result_and_metadata)

        if use_heuristic and not disable_heuristic:
            heuristic_cache.heuristic_update(result=llm_formatted_output,
                                             label=item["label"],
                                             pdf_matrix=pdf2matrix)
        
        if disable_heuristic:
            disable_heuristic = False
            use_version = version

        with open(output_json_file, "w") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()