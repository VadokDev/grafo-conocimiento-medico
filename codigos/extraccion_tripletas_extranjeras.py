import os
import json
import re
import asyncio
import csv
from collections import defaultdict
import aiohttp

from dotenv import load_dotenv

load_dotenv()

# Configuración
CSV_FILE = "wiki_medical_terms.csv"
OUTPUT_FILE = "tripletas_wiki.json"
MODELO = "gpt-4o-mini"
MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "32"))
RETRIES = 5
INITIAL_BACKOFF = 1.0
BATCH_SIZE = 10  # Guardar cada N procesados

# OpenRouter configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Relaciones organizadas por tipo de entidad
RELACIONES_MEDICAMENTOS = [
    "contraindicado_en",
    "trata",
    "uso",
    "se_administra_via",
    "otro_uso",
    "precaucion",
    "dieta",
    "olvido",
    "efecto_secundario",
    "almacenamiento",
    "sobredosis",
    "marca_comercial",
]

RELACIONES_EXAMENES = [
    "analiza",
    "alivia",
    "requiere_muestra_de",
    "indicado_si_tiene",
    "diagnostica",
    "tiene_duracion",
    "tiene_riesgo",
    "indicado_para_sintomas_de",
    "pertenece_a_categoria",
    "mide",
    "detecta",
    "es_parte_de",
    "causa_efecto_secundario",
    "utiliza_equipo",
    "se_realiza_en",
    "incluido_en",
    "util_para",
    "indica_condicion",
    "es_tipo_de",
    "ayuda_diagnosticar",
    "sugiere_condicion",
    "realizado_por",
]

RELACIONES_ENFERMEDADES = [
    "es_tipo_de",
    "presenta_sintoma",
    "diagnosticada_mediante",
    "causa_sintoma",
    "causada_por",
    "tratada_con",
    "tiene_duracion",
    "se_manifiesta_como",
    "es_hereditaria",
    "puede_causar",
    "es_congenita",
    "asociada_con",
    "afecta_organo",
    "afecta_poblacion",
    "tiene_signo",
    "prevalente_en",
    "confirmada_con",
    "es_infecciosa",
    "relacionada_con",
    "es_autoinmune",
    "mas_frecuente_en",
    "provocada_por",
    "tiene_complicacion",
    "detectada_por",
    "es_degenerativa",
    "factor_de_riesgo",
    "responde_a",
    "requiere_vacuna",
    "se_presenta_en",
]

def build_prompt(page_title: str, texto: str) -> str:
    return f"""
Analyze the following text about the medical term "{page_title}".
Extract factual information as triplets with the fields: "entidad" (entity), "relacion" (relation), and "valor" (value).

IMPORTANT: Keep entities and values in their ORIGINAL LANGUAGE (English). Only use the Spanish relation names provided below.

Valid relations are organized by category:

MEDICATIONS (if applicable):
- contraindicado_en (Contraindicated in)
- trata (Treats which conditions or diseases?)
- uso (How should this medication be used?)
- se_administra_via (Route of administration)
- otro_uso (Other uses)
- precaucion (Precautions)
- dieta (Special diet requirements)
- olvido (What to do if a dose is missed)
- efecto_secundario (Side effects)
- almacenamiento (Storage)
- sobredosis (Overdose)
- marca_comercial (Brand names)

LABORATORY TESTS (if applicable):
- analiza (What parameters or substances does this test analyze?)
- alivia (What symptoms or conditions does it relieve?)
- requiere_muestra_de (What type of sample is required?)
- indicado_si_tiene (Indicated if patient has which conditions/symptoms?)
- diagnostica (What diseases or conditions does it diagnose?)
- tiene_duracion (Duration of the test)
- tiene_riesgo (Associated risks)
- indicado_para_sintomas_de (Indicated for symptoms of)
- pertenece_a_categoria (Belongs to which category?)
- mide (What does it measure?)
- detecta (What does it detect?)
- es_parte_de (Part of which panel?)
- causa_efecto_secundario (Side effects)
- utiliza_equipo (Equipment used)
- se_realiza_en (Where is it performed?)
- incluido_en (Included in which panel?)
- util_para (Useful for)
- indica_condicion (Indicates which condition?)
- es_tipo_de (Is a type of)
- ayuda_diagnosticar (Helps diagnose)
- sugiere_condicion (Suggests which condition?)
- realizado_por (Performed by which health professional?)

DISEASES/CONDITIONS (if applicable):
- es_tipo_de (Is a type of)
- presenta_sintoma (Presents which symptom?)
- diagnosticada_mediante (Diagnosed through)
- causa_sintoma (Causes which symptom?)
- causada_por (Caused by)
- tratada_con (Treated with)
- tiene_duracion (Duration)
- se_manifiesta_como (Manifests as)
- es_hereditaria (Is hereditary - value: yes/no or details)
- puede_causar (Can cause)
- es_congenita (Is congenital - value: yes/no or details)
- asociada_con (Associated with)
- afecta_organo (Affects which organ?)
- afecta_poblacion (Affects which population?)
- tiene_signo (Has which sign?)
- prevalente_en (Prevalent in)
- confirmada_con (Confirmed with)
- es_infecciosa (Is infectious - value: yes/no or details)
- relacionada_con (Related to)
- es_autoinmune (Is autoimmune - value: yes/no or details)
- mas_frecuente_en (More frequent in)
- provocada_por (Provoked by)
- tiene_complicacion (Has which complication?)
- detectada_por (Detected by)
- es_degenerativa (Is degenerative - value: yes/no or details)
- factor_de_riesgo (Risk factor)
- responde_a (Responds to)
- requiere_vacuna (Requires vaccine - value: yes/no or vaccine name)
- se_presenta_en (Presents in)

Guidelines:
- If a category has multiple values, create a separate triplet for each value
- If a category has no information in the text, simply don't include it
- Use concise language for the value and avoid duplicates
- Keep the original language (English) for entities and values
- Only use the Spanish relation names listed above
- If you find information that fits a new relation not listed, you can include it

Text:
\"\"\"{texto}\"\"\"

Example output format:
[
  {{"entidad": "Paracetamol poisoning", "relacion": "es_tipo_de", "valor": "drug overdose"}},
  {{"entidad": "Paracetamol poisoning", "relacion": "causada_por", "valor": "excessive use of paracetamol"}},
  {{"entidad": "Paracetamol poisoning", "relacion": "presenta_sintoma", "valor": "abdominal pain"}},
  {{"entidad": "Paracetamol poisoning", "relacion": "presenta_sintoma", "valor": "nausea"}},
  {{"entidad": "Paracetamol poisoning", "relacion": "puede_causar", "valor": "liver failure"}},
  {{"entidad": "Paracetamol poisoning", "relacion": "tratada_con", "valor": "acetylcysteine"}},
  {{"entidad": "Acetylcysteine", "relacion": "trata", "valor": "paracetamol poisoning"}},
  {{"entidad": "Blood paracetamol level test", "relacion": "diagnostica", "valor": "paracetamol poisoning"}},
  {{"entidad": "Blood paracetamol level test", "relacion": "es_tipo_de", "valor": "blood test"}}
]
"""

async def call_with_retries(messages, temperature=0.2):
    """Llama a OpenRouter API con reintentos"""
    backoff = INITIAL_BACKOFF
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-kg",
        "X-Title": "Medical Knowledge Graph Extractor"
    }
    
    data = {
        "model": MODELO,
        "messages": messages,
        "temperature": temperature
    }
    
    for attempt in range(RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 429:  # Rate limit
                        if attempt == RETRIES - 1:
                            raise Exception(f"Rate limit exceeded after {RETRIES} attempts")
                        print(f"  ⏳ Rate limit, waiting {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    
                    response.raise_for_status()
                    result = await response.json()
                    return result
                    
        except asyncio.TimeoutError:
            if attempt == RETRIES - 1:
                raise
            print(f"  ⏳ Timeout, waiting {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2
            
        except aiohttp.ClientError as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  ⚠️ Client error: {e}, waiting {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2
            
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  ⚠️ Error: {e}, waiting {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2
    
    raise Exception(f"Failed after {RETRIES} attempts")

def parse_json_from_response(content: str):
    m = re.search(r'(\[.*\])', content, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
        return data
    except Exception:
        return []

async def extraer_tripletas_async(page_title: str, texto: str):
    # Limitar el texto a aproximadamente 8000 caracteres para no exceder límites del modelo
    texto_truncado = texto[:8000] if len(texto) > 8000 else texto
    
    prompt = build_prompt(page_title, texto_truncado)
    messages = [
        {"role": "system", "content": "You are an expert in extracting structured knowledge for medical knowledge graphs. Extract information maintaining the original language for entities and values."},
        {"role": "user", "content": prompt},
    ]
    resp = await call_with_retries(messages, temperature=0.2)
    # OpenRouter retorna la respuesta en el mismo formato que OpenAI
    contenido = resp['choices'][0]['message']['content'] or ""
    return parse_json_from_response(contenido)

async def procesar_entrada(sem: asyncio.Semaphore, page_title: str, texto: str, idx: int, total: int):
    async with sem:
        try:
            print(f"Processing [{idx}/{total}]: {page_title[:50]}...")
            return await extraer_tripletas_async(page_title, texto)
        except Exception as e:
            print(f"⚠️ Error processing {page_title}: {e}")
            return []

def leer_csv(csv_file: str):
    """Lee el archivo CSV y retorna una lista de (page_title, page_text)"""
    entradas = []
    
    # Aumentar el límite de tamaño de campo para manejar textos largos
    max_int = 2147483647  # Valor máximo para sistemas de 64-bit
    try:
        csv.field_size_limit(max_int)
    except OverflowError:
        # En sistemas de 32-bit, usar un valor más pequeño
        csv.field_size_limit(2147483647 // 10)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                page_title = row.get('page_title', '').strip()
                page_text = row.get('page_text', '').strip()
                if page_title and page_text:
                    entradas.append((page_title, page_text))
    except FileNotFoundError:
        print(f"❌ Error: File {csv_file} not found")
        return []
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []
    
    return entradas

async def main():
    print("🚀 Starting triplet extraction from wiki_medical_terms.csv")
    
    # Leer el CSV
    entradas = leer_csv(CSV_FILE)
    if not entradas:
        print("No entries found in CSV file.")
        return
    
    total = len(entradas)
    print(f"📊 Found {total} entries to process")
    
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # Procesar todas las entradas
    tasks = []
    for idx, (page_title, page_text) in enumerate(entradas, 1):
        task = asyncio.create_task(procesar_entrada(sem, page_title, page_text, idx, total))
        tasks.append(task)
    
    print(f"⏳ Processing with max concurrency: {MAX_CONCURRENCY}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Recolectar todas las tripletas
    all_tripletas = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"⚠️ Error in task {idx}: {res}")
            continue
        all_tripletas.extend(res)
    
    # Guardar el resultado final
    print(f"💾 Saving {len(all_tripletas)} triplets to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tripletas, f, ensure_ascii=False, indent=2)
    
    print("✅ Triplet extraction completed!")
    print(f"Total triplets extracted: {len(all_tripletas)}")
    
    # Estadísticas básicas
    relaciones_count = defaultdict(int)
    entidades_count = defaultdict(int)
    for triplet in all_tripletas:
        relaciones_count[triplet.get('relacion', 'unknown')] += 1
        entidades_count[triplet.get('entidad', 'unknown')] += 1
    
    print(f"\n📈 Statistics:")
    print(f"  - Unique relations: {len(relaciones_count)}")
    print(f"  - Unique entities: {len(entidades_count)}")
    print(f"\n🔝 Top 10 relations:")
    for rel, count in sorted(relaciones_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {rel}: {count}")

if __name__ == "__main__":
    asyncio.run(main())

