from utils.pdf2mat import PDF2Matrix
from utils.type_resolution import TypeResolver
from utils.heuristic import Heuristic
from utils.LLM import LLMExtractor

import streamlit as st
from time import time
from pathlib import Path
import os
import logging
import json
import sys
from argparse import ArgumentParser
from tqdm import tqdm


type_resolver = TypeResolver()
heuristic = Heuristic()
llm_extractor = LLMExtractor()

def logging_config():
    logger = logging.getLogger("my_logger")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logging_config()
logger = logging.getLogger("my_logger")

INPUT_DIR = Path("files")
INPUT_JSON = "dataset.json"

def run_processing(input_json_file: str):
    output_json_file = "output_results.json"
    input_files = os.listdir(INPUT_DIR)
    result_json = list()

    with open(input_json_file) as f:
        input_json = json.load(f)

    with open(output_json_file, "w") as f:
        f.write("[]")

    TEXT_BASED_VERSION = 1
    NATIVE_PDF_VERSION = 2
    disable_heuristic = False

    total = len(input_json)
    processed = 0

    for item in input_json:
        yield processed, total # Streamlit progress bar update
        processed += 1

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
            extract_form = "text_based" # Fallback to text-based extraction if matrix generation fails

        request_schema = dict(item["extraction_schema"])

        result = dict()
        if not disable_heuristic:
            result = heuristic.heuristic_preprocessing(label=item["label"], request_schema=request_schema, pdf_matrix_representation=matrix)
            heuristic_hits = list(result.keys())
            logger.info(f"Heuristic hits for {pdf_file_name}: {heuristic_hits}")

            # Remove already filled keys from the request schema
            request_schema = {k: v for k, v in request_schema.items() if k not in heuristic_hits}

            # Use a more reliable extraction form if heuristic coverage is low
            if len(heuristic_hits) / len(item["extraction_schema"]) <= 0.5:
                extract_form = NATIVE_PDF_VERSION

        start_time2 = time()
        if extract_form == TEXT_BASED_VERSION:
            logger.debug(f"Using text-based extraction for {pdf_file_name}")
            response = llm_extractor.extract_from_text_representation(input_schema=request_schema, label=item["label"], matrix=matrix, heuristic=heuristic)
        else:
            logger.debug(f"Using native PDF extraction for {pdf_file_name}")
            response = llm_extractor.extract_from_native_pdf_file(input_schema=request_schema, pdf_path=pdf_path)
        end_time2 = time()
        logger.debug(f"Extraction time for {pdf_file_name}: {end_time2 - start_time2:.2f} seconds")

        llm_formatted_output = dict(response.output_parsed)
        result.update(llm_formatted_output)
        
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.\n\n")

        result_and_metadata = {
            "pdf_path": pdf_file_name,
            "extraction_schema": result,
            "metadata": {
                "version_used": "text_based" if extract_form == TEXT_BASED_VERSION else "native_pdf",
                "latency_seconds": round(elapsed_time, 2),
                "total_tokens": response.usage.total_tokens,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_tokens": response.usage.input_tokens_details.cached_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
                "estimated_cost_usd": f"{llm_extractor.inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}"
            }
        }
        if not disable_heuristic:
            result_and_metadata["heuristic_hits"] = heuristic_hits
        result_json.append(result_and_metadata)

        if not disable_heuristic:
            heuristic.heuristic_update(result=llm_formatted_output, label=item["label"], pdf_matrix=pdf2matrix)
        
        # Reset for next iteration
        disable_heuristic = False
        extract_form = TEXT_BASED_VERSION

        with open(output_json_file, "w") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

    os.makedirs("debug_outputs", exist_ok=True)
    with open(Path("debug_outputs") / "heuristic_cache.json", "w", encoding="utf-8") as f:
        json.dump(heuristic.get_cache(), f, indent=4, ensure_ascii=False)

    yield processed, total # Final update

def streamlit_run():
    curr_dir = Path(__file__).parent.resolve()

    st.title("Extrator de PDFs")
    st.write(f"Para o correto funcionamento, certifique-se de que os arquivos PDF estejam na pasta `{curr_dir / INPUT_DIR}`..")

    input_file = st.text_input("Nome do arquivo JSON de entrada:",
                               placeholder=f"O arquivo deve estar em {curr_dir}")

    if st.button("Run"):
        progress = st.progress(0)
        status_text = st.empty()

        for processed, total in run_processing(input_file):
            percent = int((processed / total) * 100)
            progress.progress(percent)
            status_text.write(f"Processando {processed}/{total} PDFs...")

        status_text.write("✅ Processamento concluído!")
        progress.progress(100)

def build_parser():
    parser = ArgumentParser(
        description="Extrator de informações de PDFs com heurísticas e LLMs."
    )

    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warning", "error", "tqdm"],
        default="info",
        help="Nível de verbosidade do logging (default: info)."
    )

    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Executa a interface Streamlit ao invés do modo CLI."
    )

    parser.add_argument(
        "--input-json",
        type=str,
        default="dataset.json",
        help="Nome do arquivo JSON de entrada quando executado em modo CLI (default: dataset.json)."
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.verbose == "tqdm":
        logger.setLevel(100) # Disable logging
    else:
        logger.setLevel(getattr(logging, args.verbose.upper()))

    if args.streamlit:
        streamlit_run()
    else:
        # Progress bar for CLI mode
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
        total_files = len(files)
        for processed, total in tqdm(run_processing(args.input_json), total=total_files, desc="Processing PDFs", unit="file", ncols=100):
            pass

if __name__ == "__main__":
    main()