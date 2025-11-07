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
import pandas as pd
import plotly.express as px
from glob import glob
from datetime import datetime


type_resolver = TypeResolver()
heuristic = Heuristic()
llm_extractor = LLMExtractor()

def logging_config():
    logger = logging.getLogger("my_logger")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logging_config()
logger = logging.getLogger("my_logger")

INPUT_DIR = Path("files") # Where PDFs are stored

def run_processing(input_json_path: str):
    if not input_json_path or not os.path.isfile(input_json_path):
        logger.error(f"Input JSON file '{input_json_path}' does not exist.")
        raise Exception("JSON inválido.")
    
    input_files = os.listdir(INPUT_DIR)
    if len(input_files) == 0:
        logger.error(f"No PDF files found in input directory '{INPUT_DIR}'.")
        raise Exception(f"Nenhum PDF encontrado em {INPUT_DIR}.")

    try:
        with open(input_json_path) as f:
            input_json = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Couldn't load {input_json_path}.")
        raise Exception(f"Erro ao ler o JSON de entrada '{input_json_path}'.")
    
    total = len(input_json)
    processed = 0
    if total == 0:
        logger.error(f"No data to process in JSON {input_json_path}.")
        raise Exception(f"Nenhum dado a ser processado no JSON {input_json_path}.")

    result_json = list()
    time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    output_json_path = f"results_{time_stamp}.json"
    with open(output_json_path, "w") as f:
        f.write("[]")

    TEXT_BASED_VERSION = 1
    NATIVE_PDF_VERSION = 2
    disable_heuristic = False
    extract_form = NATIVE_PDF_VERSION # Start with native PDF extraction


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
        heuristic_hits = list()
        if not disable_heuristic:
            result = heuristic.heuristic_preprocessing(label=item["label"], request_schema=request_schema, pdf_matrix_representation=matrix)
            heuristic_hits = list(result.keys())
            logger.info(f"Heuristic hits for {pdf_file_name}: {heuristic_hits}")

            # Remove already filled keys from the request schema
            request_schema = {k: v for k, v in request_schema.items() if k not in heuristic_hits}

            # Use a more reliable extraction form if heuristic coverage is low
            if len(heuristic_hits) / len(item["extraction_schema"]) <= 0.5:
                extract_form = NATIVE_PDF_VERSION

        # If heuristic didn't fill all keys, proceed with LLM extraction
        if len(heuristic_hits) != len(item["extraction_schema"]):
            if extract_form == TEXT_BASED_VERSION:
                logger.debug(f"Using text-based extraction for {pdf_file_name}")
                response = llm_extractor.extract_from_text_representation(input_schema=request_schema, label=item["label"], matrix=matrix, heuristic=heuristic)
            else:
                logger.debug(f"Using native PDF extraction for {pdf_file_name}")
                response = llm_extractor.extract_from_native_pdf_file(input_schema=request_schema, pdf_path=pdf_path)

            llm_formatted_output = dict(response.output_parsed)
            result.update(llm_formatted_output)

            if not disable_heuristic:
                heuristic.heuristic_update(result=llm_formatted_output, label=item["label"], pdf_matrix=pdf2matrix)
        else:
            logger.info(f"All keys extracted via heuristic for {pdf_file_name}. Skipping LLM extraction.")
            
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Processed {pdf_file_name} in {elapsed_time:.2f} seconds.\n\n")

        result_and_metadata = {
            "extraction_schema": result,
            "metadata": {
                "pdf_path": pdf_file_name,
                "label": item["label"],
                "version_used": "text_based" if extract_form == TEXT_BASED_VERSION else "native_pdf",
                "latency_seconds": round(elapsed_time, 2),
                "total_tokens": response.usage.total_tokens,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_tokens": response.usage.input_tokens_details.cached_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
                "estimated_cost_usd": f"{llm_extractor.inference_cost_estimation(response.usage.input_tokens, response.usage.output_tokens):3e}",
                "heuristic_hits": heuristic_hits
            }
        }

        result_json.append(result_and_metadata)
        
        # Reset for next iteration
        disable_heuristic = False
        extract_form = TEXT_BASED_VERSION

        with open(output_json_path, "w") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

    os.makedirs("debug_outputs", exist_ok=True)
    with open(Path("debug_outputs") / "heuristic_cache.json", "w", encoding="utf-8") as f:
        json.dump(heuristic.get_cache(), f, indent=4, ensure_ascii=False)

    yield processed, total # Final update

def streamlit_run():
    curr_dir = Path(__file__).parent.resolve()

    st.title("Extrator de PDFs")
    st.write(f"Para o correto funcionamento, certifique-se de que os arquivos PDF referenciados pelo JSON de entrada estejam na pasta `{curr_dir / INPUT_DIR}`.")

    default_path = os.path.join(curr_dir, "*.json")
    existing_files = glob(default_path)

    st.header("📥 Selecionar Arquivo de Entrada")   

    st.write(f"Apenas arquivos em `{curr_dir}` serão considerados.")
    input_json_path = st.selectbox("Escolha um arquivo:", existing_files,
                                accept_new_options=False,
                                index=None,
                                placeholder="Selecione um arquivo...")

    if input_json_path is None or not os.path.isfile(input_json_path):
        st.info("Por favor, selecione um arquivo JSON válido para continuar.")
    else:
        st.success(f"✅ Arquivo carregado: {os.path.basename(input_json_path)}")

        if st.button("Run"):
            progress = st.progress(0)
            status_text = st.empty()

            try:
                for processed, total in run_processing(input_json_path):
                    percent = int((processed / total) * 100)
                    progress.progress(percent)
                    status_text.write(f"Processando {processed}/{total} PDFs...")   
            except Exception as e:
                st.error(f"Ocorreu um erro durante o processamento. \n\n Erro: {e}")

            status_text.write("✅ Processamento concluído!")
            progress.progress(100)
    
    if st.button("Show stats"):
        # Take the latest results file
        results_files = sorted(glob("results_*.json"), reverse=True)
        if len(results_files) == 0:
            st.error("Nenhum arquivo de resultados encontrado. Execute o processamento primeiro.")
            return
        results_json = results_files[0]
        st.write(f"Carregando resultados de: `{results_json}`")

        try:
            with open(results_json) as f:
                results_data = json.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar resultados: {e}")
            return

        if len(results_data) == 0:
            st.error("Arquivo com os resultados e estatísticas vazio.")
            return

        stats = []
        for item in results_data:
            d = {"num_keys_extracted": len(item["extraction_schema"])}
            d.update(item["metadata"])
            d["estimated_cost_usd"] = float(d["estimated_cost_usd"])
            d["heuristic_hits_percent"] = (len(d["heuristic_hits"]) / d["num_keys_extracted"]) * 100 if d["num_keys_extracted"] > 0 else 0
            d.pop("heuristic_hits")  # Remove detailed heuristic hits for stats 
            stats.append(d)

        df = pd.DataFrame(stats)

        # ======= Overall statistics
        st.header("✅ Estatísticas Gerais da Extração")

        col1, col2, col3 = st.columns(3)
        col1.metric("PDFs Processados", len(df))
        col2.metric("Custo Total (USD)", f"${df['estimated_cost_usd'].sum():.8f}")
        col3.metric("Número Total de Chaves Extraídas", df["num_keys_extracted"].sum())

        col4, col5, col6 = st.columns(3)
        col4.metric("Aproveitamento Heurística (%)", f"{df['heuristic_hits_percent'].mean():.2f}%")
        col5.metric("Latência Média (s)", f"{df['latency_seconds'].mean():.2f}")
        col6.metric("Média de Tokens Totais", f"{df['total_tokens'].mean():.2f}")

        # ======= Relationships between variables
        st.header("📊 Análises Gerais")
        st.dataframe(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tokens x Latência", "Performance Heurística x Latência", "Performance Heurística x Tokens", "Versão Prompt x Tokens", "Versão Prompt x Latência"])

        with tab1:
            fig = px.scatter(
                df,
                x="total_tokens",
                y="latency_seconds",
                color="label",
                hover_data=["label", "pdf_path", "num_keys_extracted", "estimated_cost_usd", "heuristic_hits_percent"],
                title="Relação entre Tokens e Latência (por PDF)",
            )
            st.plotly_chart(fig)
        with tab2:
            fig = px.scatter(
                df,
                x="heuristic_hits_percent",
                y="latency_seconds",
                color="label",
                hover_data=["label", "pdf_path", "num_keys_extracted", "estimated_cost_usd", "total_tokens"],
                title="Relação entre Performance da Heurística e Latência (por PDF)",
            )
            st.plotly_chart(fig)
        with tab3:
            fig = px.scatter(
                df,
                x="heuristic_hits_percent",
                y="total_tokens",
                color="label",
                hover_data=["label", "pdf_path", "num_keys_extracted", "estimated_cost_usd", "latency_seconds"],
                title="Relação entre Performance da Heurística e Tokens (por PDF)",
            )
            st.plotly_chart(fig)
        with tab4:
            fig = px.box(
                df,
                x="version_used",
                y="total_tokens",
                points="all",
                title="Distribuição de Tokens por Versão de Prompt Usada (ver README)",
            )
            st.plotly_chart(fig)
        with tab5:
            fig = px.box(
                df,
                x="version_used",
                y="latency_seconds",
                points="all",
                title="Distribuição de Latência por Versão de Prompt Usada (ver README)",
            )
            st.plotly_chart(fig)


        # ======= Statistics by label
        st.header("📌 Estatísticas por Label")

        df_labels = (
            df.groupby("label")
            .agg(
                num_pdfs=("label", "count"),
                avg_num_keys_extracted=("num_keys_extracted", "mean"),
                total_estimated_cost_usd=("estimated_cost_usd", "sum"),
                avg_heuristic_hits_percent=("heuristic_hits_percent", "mean"),
                avg_latency_seconds=("latency_seconds", "mean"),
                avg_total_tokens=("total_tokens", "mean"),
            )
            .reset_index()
        )

        st.dataframe(df_labels)

        st.subheader("📈 Visualizações Interativas")

        tab1, tab2, tab3, tab4 = st.tabs(["Custo Total", "Latência", "Tokens Totais", "Performance Heurística"])

        with tab1:
            fig = px.bar(df_labels, x="label", y="total_estimated_cost_usd", title="Custo Total por Label")
            st.plotly_chart(fig)

        with tab2:
            fig = px.box(df, x="label", y="latency_seconds", points="all",
                        title="Distribuição de Latência por Label")
            st.plotly_chart(fig)

        with tab3:
            fig = px.box(df, x="label", y="total_tokens", points="all",
                            title="Distribuição de Tokens")
            st.plotly_chart(fig)
        with tab4:
            fig = px.bar(df_labels, x="label", y="avg_heuristic_hits_percent", title="Performance Média da Heurística por Label")
            st.plotly_chart(fig)

def build_parser():
    parser = ArgumentParser(
        description="Extrator de informações de PDFs com heurísticas e LLMs."
    )

    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warning", "error", "tqdm"],
        default="info",
        help="Nível de verbosidade do logging quando executado em modo CLI (default: info)."
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

    if args.streamlit:
        logger.setLevel(100)
        streamlit_run()
    elif args.verbose == "tqdm":
        logger.setLevel(100)  # Suppress logging when using tqdm
        
        with open(args.input_json) as f:
            input_json = json.load(f)
        total_files = len(input_json)

        for processed, total in tqdm(run_processing(args.input_json), total=total_files, desc="Processing PDFs", unit="file", ncols=100):
            pass
    else:
        logging_dict = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        logger.setLevel(logging_dict.get(args.verbose, logging.INFO))
        for _, _ in run_processing(args.input_json):
            pass
            
if __name__ == "__main__":
    main()