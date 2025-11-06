"""
LLM-related utilities for information extraction from PDFs.
"""

from utils.heuristic import Heuristic

from openai import OpenAI
import os
from pathlib import Path
from pydantic import create_model
from typing import Optional
import yaml
import logging
import base64
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("my_logger")
PRICE_PER_1M_INPUT_TOKENS = 0.25 # USD (02/11/2025)
PRICE_PER_1M_OUTPUT_TOKENS = 2.0 # USD (02/11/2025)

TEXT_BASED_EXTRACTION_PROMPT = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no conteúdo do PDF. O conteúdo do PDF é fornecido por meio de uma representação estruturada em forma de matriz, mantendo uma estrutura similar à disposição visual do PDF.
Alguns campos do YAML possuem exemplos de valores extraídos previamente. Utilize esses exemplos para guiar a extração, buscando padrões similares na representação do PDF.

# Instruções de Extração
## PRECISÃO E COMPLETUDE
- Extraia apenas informações explicitamente presentes no PDF
- Não infira ou invente valores não presentes
- Utiliza as descrições dos campos no YAML para guiar a extração, em alguns casos os valores possíveis para um campo estão explicitamente descritos no YAML, não ignore essas descrições
- Utilize os exemplos fornecidos no YAML para identificar padrões, mas não copie valores diretamente a menos que estejam presentes no PDF
- Não ignore informações relevantes que correspondam aos campos
- Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no texto do PDF. Se não estiver explicitamente presente ou reconhecível, não use termos similares para preencher a chave. Na dúvida, prefira deixar o campo como null
- Para campos ausentes, use `null`, não use strings vazias

# YAML de requisição
{request_yaml}

# Representação estruturada do conteúdo PDF (matriz):
{pdf_matrix_representation}

Comece.
"""

NATIVE_PDF_EXTRACTION_PROMPT = """
# Tarefa
Sua tarefa é retornar um json, e somente um json, preenchendo os campos solicitados por meio do YAML de requisição abaixo com base no PDF.

# Instruções de Extração
## PRECISÃO E COMPLETUDE
- Extraia apenas informações explicitamente presentes no PDF
- Não infira ou invente valores não presentes
- Utiliza as descrições dos campos no YAML para guiar a extração, em alguns casos os valores possíveis para um campo estão explicitamente descritos no YAML, não ignore essas descrições
- Não ignore informações relevantes que correspondam aos campos
- Preencha uma chave **somente se** o nome da chave ou uma abreviação **claramente correspondente** estiver presente no texto do PDF. Se não estiver explicitamente presente ou reconhecível, não use termos similares para preencher a chave. Na dúvida, prefira deixar o campo como null
- Para campos ausentes, use `null`, não use strings vazias

# YAML de requisição
{request_yaml}

Comece.
"""

class LLMExtractor:
    def __init__(self):
        self.__client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def inference_cost_estimation(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
        output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
        total_cost = input_cost + output_cost
        return total_cost

    def extract_from_text_representation(self, input_schema: dict, label: str, matrix: list, heuristic: Heuristic):
        """
        Extract information by passing the text representation of the PDF (matrix form) to the model.
        Normally, it's cheaper and faster than passing the native PDF.
        """
        processed_schema = dict()
        for key, value in input_schema.items():
            processed_schema[key] = {"descricao": value}
            examples = heuristic.get_examples_for_key(key, label)
            if examples:
                processed_schema[key]["examples"] = examples

        mat_to_str = [f"Row {i+1}: " + " | ".join(row) for i, row in enumerate(matrix)]
        mat_to_str = "\n".join(mat_to_str)

        os.makedirs("debug_outputs", exist_ok=True)
        with open(Path("debug_outputs") / "pdf_representation.txt", "a", encoding="utf-8") as f:
            f.write(mat_to_str + "\n\n" + ("="*80) + "\n\n")

        output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
        OutputModelStructure = create_model("OutputModelStructure", **output_structure)

        yaml_schema = yaml.dump(processed_schema, allow_unicode=True)

        prompt = TEXT_BASED_EXTRACTION_PROMPT.format(request_yaml=yaml_schema, pdf_matrix_representation=mat_to_str).strip()

        # logger.debug(f"Prompt: {prompt}")

        response = self.__client.responses.parse(model="gpt-5-mini-2025-08-07",
                                                text_format=OutputModelStructure,
                                                reasoning={"effort":"minimal"},
                                                input=prompt)
        return response
        
    def extract_from_native_pdf_file(self, input_schema: dict, pdf_path: str):
        """
        Extract information by passing the native PDF file to the model.
        This method may be more accurate but is generally more expensive and slower.
        """

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode()

        output_structure = {key: (Optional[str], None) for key in input_schema.keys()}
        OutputModelStructure = create_model("OutputModelStructure", **output_structure)

        yaml_schema = yaml.dump(input_schema, allow_unicode=True)

        prompt = NATIVE_PDF_EXTRACTION_PROMPT.format(request_yaml=yaml_schema).strip()

        response = self.__client.responses.parse(model="gpt-5-mini-2025-08-07",
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