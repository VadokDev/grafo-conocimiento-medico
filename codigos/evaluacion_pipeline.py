# -*- coding: utf-8 -*-
"""
Medical Knowledge Graph Evaluation Pipeline
Automatically processes triplets, generates questions, and evaluates multiple models
"""

import json
import os
import time
import threading
import signal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to evaluate
MODELS = {
    "llama": "meta-llama/llama-3.1-8b-instruct",
    "qwen": "qwen/qwen-2.5-7b-instruct",
    "ministral": "mistralai/ministral-8b"
}

# Evaluation model
EVALUATOR_MODEL = "gpt-4o-mini"

# Max workers for parallel processing
MAX_WORKERS = 15

# ============================================================================
# QUESTION MAPPINGS (IN ENGLISH)
# ============================================================================

QUESTION_MAPPING_MEDICATIONS = {
    "contraindicado_en": "What conditions or situations is {entity} contraindicated in?",
    "trata": "What conditions or diseases does {entity} treat?",
    "uso": "How should {entity} be used?",
    "se_administra_via": "What is the route of administration for {entity}?",
    "otro_uso": "What are other uses of {entity}?",
    "precaucion": "What precautions should be taken with {entity}?",
    "dieta": "What dietary requirements are there for {entity}?",
    "olvido": "What should be done if a dose of {entity} is missed?",
    "efecto_secundario": "What are the side effects of {entity}?",
    "almacenamiento": "How should {entity} be stored?",
    "sobredosis": "What happens in case of {entity} overdose?",
    "marca_comercial": "What are the brand names for {entity}?",
}

QUESTION_MAPPING_LAB_TESTS = {
    "analiza": "What parameters or substances does {entity} analyze?",
    "alivia": "What symptoms or conditions does {entity} relieve?",
    "requiere_muestra_de": "What type of sample does {entity} require?",
    "indicado_si_tiene": "When is {entity} indicated?",
    "diagnostica": "What diseases or conditions does {entity} diagnose?",
    "tiene_duracion": "What is the duration of {entity}?",
    "tiene_riesgo": "What are the risks associated with {entity}?",
    "indicado_para_sintomas_de": "What symptoms is {entity} indicated for?",
    "pertenece_a_categoria": "What category does {entity} belong to?",
    "mide": "What does {entity} measure?",
    "detecta": "What does {entity} detect?",
    "es_parte_de": "What panel or study is {entity} part of?",
    "causa_efecto_secundario": "What side effects can {entity} cause?",
    "utiliza_equipo": "What equipment does {entity} use?",
    "se_realiza_en": "Where is {entity} performed?",
    "incluido_en": "What panel or profile includes {entity}?",
    "util_para": "What is {entity} useful for?",
    "indica_condicion": "What condition does {entity} indicate?",
    "es_tipo_de": "What type of test is {entity}?",
    "ayuda_diagnosticar": "What does {entity} help diagnose?",
    "sugiere_condicion": "What condition does {entity} suggest?",
    "realizado_por": "Who performs {entity}?",
}

QUESTION_MAPPING_DISEASES = {
    "es_tipo_de": "What type of condition is {entity}?",
    "presenta_sintoma": "What symptoms does {entity} present?",
    "diagnosticada_mediante": "How is {entity} diagnosed?",
    "causa_sintoma": "What symptoms does {entity} cause?",
    "causada_por": "What causes {entity}?",
    "tratada_con": "How is {entity} treated?",
    "tiene_duracion": "What is the duration of {entity}?",
    "se_manifiesta_como": "How does {entity} manifest?",
    "es_hereditaria": "Is {entity} hereditary?",
    "puede_causar": "What can {entity} cause?",
    "es_congenita": "Is {entity} congenital?",
    "asociada_con": "What is {entity} associated with?",
    "afecta_organo": "What organs does {entity} affect?",
    "afecta_poblacion": "What populations does {entity} affect?",
    "tiene_signo": "What signs does {entity} have?",
    "prevalente_en": "Where or in whom is {entity} prevalent?",
    "confirmada_con": "How is {entity} confirmed?",
    "es_infecciosa": "Is {entity} infectious?",
    "relacionada_con": "What is {entity} related to?",
    "es_autoinmune": "Is {entity} autoimmune?",
    "mas_frecuente_en": "Who is {entity} more frequent in?",
    "provocada_por": "What provokes {entity}?",
    "tiene_complicacion": "What complications does {entity} have?",
    "detectada_por": "How is {entity} detected?",
    "es_degenerativa": "Is {entity} degenerative?",
    "factor_de_riesgo": "What are the risk factors for {entity}?",
    "responde_a": "What does {entity} respond to?",
    "requiere_vacuna": "Does {entity} require a vaccine?",
    "se_presenta_en": "Where does {entity} present?",
}

# ============================================================================
# STEP 1: GENERATE QUESTIONS FROM TRIPLETS
# ============================================================================

def transform_triplets_to_qa(triplets: List[Dict], question_mappings: List[Dict]) -> List[Dict]:
    """
    Transforms triplets into question-answer pairs.
    Groups all answers that share the same entity and relation.
    
    Args:
        triplets: List of triplets with structure {entidad, relacion, valor}
        question_mappings: List of question mapping dictionaries to try
        
    Returns:
        List of Q&A pairs with structure {question, answer}
    """
    # Merge all question mappings
    merged_mapping = {}
    for mapping in question_mappings:
        merged_mapping.update(mapping)
    
    # Dictionary to group answers by entity and relation
    grouped = {}
    
    for triplet in triplets:
        entity = triplet.get("entidad", "")
        relation = triplet.get("relacion", "")
        value = triplet.get("valor", "")
        
        if not entity or not relation or not value:
            continue
        
        # Create unique key for entity + relation
        key = (entity, relation)
        
        # Add value to the list of answers
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(value)
    
    # Generate question-answer pairs
    qa_pairs = []
    
    for (entity, relation), values in grouped.items():
        # Get the question template
        if relation in merged_mapping:
            template = merged_mapping[relation]
            question = template.format(entity=entity)
            # Join all values with comma
            answer = ", ".join(values)
            
            qa_pairs.append({
                "question": question,
                "answer": answer
            })
    
    return qa_pairs


def generate_questions_from_file(input_file: str, output_file: str, question_mappings: List[Dict]):
    """
    Reads a JSON file with triplets and generates a JSON file with Q&A pairs.
    
    Args:
        input_file: Path to JSON file with triplets
        output_file: Path to save the Q&A pairs JSON
        question_mappings: List of question mapping dictionaries
    """
    print(f"\n{'='*80}")
    print(f"📚 GENERATING QUESTIONS FROM: {input_file}")
    print(f"{'='*80}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File {input_file} does not exist")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            triplets = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    if not isinstance(triplets, list):
        print(f"❌ Error: File must contain a list of triplets")
        return
    
    print(f"✓ Loaded {len(triplets)} triplets")
    
    qa_pairs = transform_triplets_to_qa(triplets, question_mappings)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated {len(qa_pairs)} question-answer pairs")
    print(f"✓ Saved to: {output_file}")


# ============================================================================
# STEP 2: GENERATE MODEL RESPONSES
# ============================================================================

class ResponseGenerator:
    """Generates responses to questions using OpenRouter API"""
    
    def __init__(self, api_key: str, model: str, max_workers: int = 10):
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.write_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.processed_count = 0
        self.output_file = None
        self.is_first_write = True
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # High-quality system prompt for medical Q&A
        self.SYSTEM_PROMPT = """You are a medical knowledge expert. Answer the following medical questions accurately and concisely. 
Provide only the requested information without unnecessary explanations. Your answers should be factual, precise, and based on established medical knowledge."""
    
    def signal_handler(self, signum, frame):
        """Handles Ctrl+C"""
        print("\n⚠️  Stopping process...")
        self.stop_flag.set()
    
    def generate_response(self, question: str, max_retries: int = 3) -> Optional[str]:
        """Generates a response using OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/medical-evaluation",
            "X-Title": "Medical Knowledge Evaluation"
        }
        
        for attempt in range(max_retries):
            if self.stop_flag.is_set():
                return None
            
            try:
                data = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
                
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                answer = result['choices'][0]['message']['content'].strip()
                return answer
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5
                    print(f"  ⏳ Timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Timeout after {max_retries} attempts")
                    return None
            
            except requests.exceptions.HTTPError as e:
                status_code = response.status_code if 'response' in locals() else None
                
                if status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 5
                        print(f"  ⏳ Rate limit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ Rate limit exceeded")
                        return None
                else:
                    print(f"  ❌ HTTP error {status_code}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return None
            
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def process_question(self, item: Dict, index: int, total: int) -> Optional[Dict]:
        """Processes a single question"""
        if self.stop_flag.is_set():
            return None
        
        question = item.get("question", "")
        expected_answer = item.get("answer", "")
        
        print(f"\n  📋 Question {index + 1}/{total}: {question[:60]}{'...' if len(question) > 60 else ''}")
        
        if not question:
            print(f"  ⚠️  Empty question")
            return None
        
        model_answer = self.generate_response(question)
        
        if model_answer is None:
            print(f"  ❌ No response generated")
            return None
        
        print(f"  ✅ Response generated ({len(model_answer)} chars)")
        
        return {
            "id": index + 1,
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "processing_date": datetime.now().isoformat()
        }
    
    def write_result_to_json(self, result: Dict):
        """Writes result in a thread-safe manner"""
        with self.write_lock:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    if not self.is_first_write:
                        f.write(',\n')
                    else:
                        self.is_first_write = False
                    
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)
                    indented_json = '\n'.join(['  ' + line for line in json_str.split('\n')])
                    f.write(indented_json)
                    f.flush()
            except Exception as e:
                print(f"❌ Error writing: {e}")
    
    def close_json_file(self):
        """Closes the JSON file correctly"""
        with self.write_lock:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write('\n]')
                    f.flush()
            except Exception as e:
                print(f"❌ Error closing file: {e}")
    
    def process_questions_file(self, input_file: str, output_file: str):
        """Processes all questions from a JSON file"""
        
        print(f"\n{'='*80}")
        print(f"🤖 GENERATING RESPONSES")
        print(f"{'='*80}")
        print(f"📄 Input: {input_file}")
        print(f"💾 Output: {output_file}")
        print(f"🤖 Model: {self.model}")
        print(f"⚙️  Workers: {self.max_workers}")
        
        if not os.path.exists(input_file):
            print(f"❌ File {input_file} does not exist")
            return
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
        
        if not isinstance(questions, list):
            print(f"❌ File must contain a list of questions")
            return
        
        if not questions:
            print(f"⚠️  No questions in file")
            return
        
        self.output_file = output_file
        self.processed_count = 0
        self.is_first_write = True
        
        total_questions = len(questions)
        print(f"🚀 Processing {total_questions} questions\n")
        
        # Initialize output JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            f.flush()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {
                    executor.submit(self.process_question, item, idx, total_questions): idx
                    for idx, item in enumerate(questions)
                }
                
                for future in as_completed(future_to_index):
                    if self.stop_flag.is_set():
                        print("\n⏸️  Cancelling...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        result = future.result(timeout=120)
                        
                        if result is not None:
                            self.write_result_to_json(result)
                            self.processed_count += 1
                            progress = (self.processed_count / total_questions) * 100
                            print(f"  💾 Saved {self.processed_count}/{total_questions} ({progress:.1f}%)")
                    
                    except Exception as e:
                        index = future_to_index.get(future, "?")
                        print(f"❌ Error in question {index + 1}: {e}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Manual interruption (Ctrl+C)")
            self.stop_flag.set()
        
        finally:
            self.close_json_file()
            
            print(f"\n{'='*80}")
            if self.stop_flag.is_set():
                print(f"⏸️  Process stopped")
            else:
                print(f"🎉 Completed!")
            print(f"✅ Processed: {self.processed_count}/{total_questions}")
            print(f"📄 File: {output_file}")
            print(f"{'='*80}")


# ============================================================================
# STEP 3: EVALUATE MODEL RESPONSES
# ============================================================================

class ResponseEvaluator:
    """Evaluates model responses against expected answers"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_workers: int = 10):
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.stop_flag = threading.Event()
        self.write_lock = threading.Lock()
        self.is_first_write = True
        self.output_file = None
        self.processed_count = 0
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # High-quality evaluation prompt
        self.SYSTEM_PROMPT = """You are an expert medical knowledge evaluator. Your task is to compare a model's answer with the expected answer and determine if the model answered correctly.

Respond ONLY with "correct" or "incorrect" (no justifications or explanations).

Evaluation criteria:
1. The model answer correctly addresses the main intent of the question
2. The formulation can be considered equivalent even if not textually identical
3. The medical facts presented are accurate and relevant
4. Minor differences in wording are acceptable if the meaning is preserved"""
    
    def signal_handler(self, signum, frame):
        print("\n⚠️  Stopping process...")
        self.stop_flag.set()
    
    def evaluate_response(self, question: str, expected_answer: str, model_answer: str, max_retries: int = 3):
        """Evaluates a response using the API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Question: {question}

Expected answer: {expected_answer}

Model answer: {model_answer}

Is the model answer correct according to the expected answer?
Respond only with 'correct' or 'incorrect'."""
        
        for attempt in range(max_retries):
            if self.stop_flag.is_set():
                return None
            
            try:
                response = requests.post(OPENROUTER_URL, headers=headers, json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10,
                    "temperature": 0
                })
                
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
                    print(f"  ⏳ Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                text = result["choices"][0]["message"]["content"].strip().lower()
                
                if "incorrect" in text:
                    return "incorrect"
                elif "correct" in text:
                    return "correct"
                else:
                    return "indeterminate"
            
            except Exception as e:
                print(f"  ❌ Error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        
        return "error"
    
    def process_item(self, item: Dict, idx: int, total: int):
        if self.stop_flag.is_set():
            return None
        
        print(f"\n  📋 Evaluating {idx + 1}/{total}")
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        model_answer = item.get("model_answer", "")
        
        evaluation = self.evaluate_response(question, expected_answer, model_answer)
        print(f"  ✅ Evaluation: {evaluation}")
        
        item["evaluation"] = evaluation
        item["evaluation_date"] = datetime.now().isoformat()
        return item
    
    def write_result_to_json(self, result: Dict):
        with self.write_lock:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    if not self.is_first_write:
                        f.write(',\n')
                    else:
                        self.is_first_write = False
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)
                    indented_json = '\n'.join(['  ' + line for line in json_str.split('\n')])
                    f.write(indented_json)
                    f.flush()
            except Exception as e:
                print(f"❌ Error writing: {e}")
    
    def close_json_file(self):
        with self.write_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write('\n]')
                f.flush()
    
    def evaluate_file(self, input_file: str, output_file: str):
        """Evaluates all responses from a file"""
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUATING RESPONSES")
        print(f"{'='*80}")
        print(f"📄 Input: {input_file}")
        print(f"💾 Output: {output_file}")
        print(f"🤖 Evaluator: {self.model}")
        
        if not os.path.exists(input_file):
            print(f"❌ File {input_file} not found")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ File must be a list of JSON objects")
            return
        
        total = len(data)
        self.output_file = output_file
        self.is_first_write = True
        self.processed_count = 0
        
        print(f"🚀 Evaluating {total} responses\n")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            f.flush()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_item, item, i, total): i for i, item in enumerate(data)}
                
                for future in as_completed(futures):
                    if self.stop_flag.is_set():
                        print("\n⏸️  Cancelling...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    result = future.result()
                    if result:
                        self.write_result_to_json(result)
                        self.processed_count += 1
                        progress = (self.processed_count / total) * 100
                        print(f"  💾 Saved {self.processed_count}/{total} ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            print("\n⚠️ Manual interruption")
            self.stop_flag.set()
        finally:
            self.close_json_file()
            print(f"\n✅ Evaluation completed ({self.processed_count}/{total})")
            print(f"💾 File saved: {output_file}")
            print(f"{'='*80}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline: generates questions, model responses, and evaluations"""
    
    print("\n" + "="*80)
    print("🏥 MEDICAL KNOWLEDGE GRAPH EVALUATION PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Generate questions from triplet datasets")
    print("2. Generate responses from 3 models")
    print("3. Evaluate all responses")
    print("\nDatasets: dataset_drugs_tripletas.json, tripletas_wiki.json")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Evaluator: {EVALUATOR_MODEL}")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: GENERATE QUESTIONS
    # ========================================================================
    
    print("\n" + "🔹"*40)
    print("STEP 1: GENERATING QUESTIONS")
    print("🔹"*40)
    
    # Generate questions for drugs dataset (only medication relations)
    generate_questions_from_file(
        input_file="dataset_drugs_tripletas.json",
        output_file="preguntas_dataset_drugs.json",
        question_mappings=[QUESTION_MAPPING_MEDICATIONS]
    )
    
    # Generate questions for wiki dataset (all relation types)
    generate_questions_from_file(
        input_file="tripletas_wiki.json",
        output_file="preguntas_dataset_wiki.json",
        question_mappings=[
            QUESTION_MAPPING_MEDICATIONS,
            QUESTION_MAPPING_LAB_TESTS,
            QUESTION_MAPPING_DISEASES
        ]
    )
    
    # ========================================================================
    # STEP 2: GENERATE MODEL RESPONSES
    # ========================================================================
    
    print("\n" + "🔹"*40)
    print("STEP 2: GENERATING MODEL RESPONSES")
    print("🔹"*40)
    
    datasets = [
        ("preguntas_dataset_drugs.json", "drugs"),
        ("preguntas_dataset_wiki.json", "wiki")
    ]
    
    for questions_file, dataset_name in datasets:
        if not os.path.exists(questions_file):
            print(f"⚠️  Skipping {questions_file} (not found)")
            continue
        
        for model_name, model_id in MODELS.items():
            output_file = f"respuestas_{dataset_name}_{model_name}.json"
            
            print(f"\n{'─'*80}")
            print(f"Processing: {dataset_name} with {model_name}")
            print(f"{'─'*80}")
            
            generator = ResponseGenerator(
                api_key=API_KEY,
                model=model_id,
                max_workers=MAX_WORKERS
            )
            
            generator.process_questions_file(questions_file, output_file)
    
    # ========================================================================
    # STEP 3: EVALUATE RESPONSES
    # ========================================================================
    
    print("\n" + "🔹"*40)
    print("STEP 3: EVALUATING RESPONSES")
    print("🔹"*40)
    
    for questions_file, dataset_name in datasets:
        for model_name, model_id in MODELS.items():
            responses_file = f"respuestas_{dataset_name}_{model_name}.json"
            evaluations_file = f"evaluaciones_{dataset_name}_{model_name}.json"
            
            if not os.path.exists(responses_file):
                print(f"⚠️  Skipping {responses_file} (not found)")
                continue
            
            print(f"\n{'─'*80}")
            print(f"Evaluating: {dataset_name} - {model_name}")
            print(f"{'─'*80}")
            
            evaluator = ResponseEvaluator(
                api_key=API_KEY,
                model=EVALUATOR_MODEL,
                max_workers=MAX_WORKERS
            )
            
            evaluator.evaluate_file(responses_file, evaluations_file)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED!")
    print("="*80)
    print("\n📊 Generated files:")
    print("\nQuestion files:")
    print("  - preguntas_dataset_drugs.json")
    print("  - preguntas_dataset_wiki.json")
    print("\nResponse files:")
    for dataset_name in ["drugs", "wiki"]:
        for model_name in MODELS.keys():
            print(f"  - respuestas_{dataset_name}_{model_name}.json")
    print("\nEvaluation files:")
    for dataset_name in ["drugs", "wiki"]:
        for model_name in MODELS.keys():
            print(f"  - evaluaciones_{dataset_name}_{model_name}.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

