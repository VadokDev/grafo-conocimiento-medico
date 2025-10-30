# Grafo de Conocimiento Médico Latinoamericano

**Autores:** Sebastián Burgos, Gonzalo Fernández, Nicolás Sumonte  
**Institución:** Pontificia Universidad Católica de Chile  
**Fecha:** Octubre 2025  
**Repositorio:** https://github.com/VadokDev/grafo-conocimiento-medico

---

## Resumen

Dataset cultural de **78.208 tripletas** sobre conocimiento médico latinoamericano para evaluar brechas regionales en modelos de lenguaje. Incluye 76.234 tripletas adicionales de fuentes estadounidenses para comparación.

**Hallazgo Principal:** Todos los modelos evaluados muestran bajo desempeño sistemático en contenido médico latinoamericano (17-31% vs 30-54% en contenido universal), evidenciando subrepresentación del conocimiento regional.

---

## Dataset

### Estructura de Archivos

```
publicacion/
├── datos/                          # Datasets y evaluaciones
│   ├── dataset_drugs_tripletas.json      # 45.158 tripletas (medicamentos)
│   ├── dataset_examenes_lab.json         # 4.315 tripletas (exámenes)
│   ├── tripletas_wiki.json               # 13.391 tripletas (enfermedades)
│   ├── preguntas_dataset_*.json          # Preguntas generadas
│   └── respuestas_*.json                 # Respuestas de modelos
├── evaluaciones/                   # Resultados de evaluación (9 archivos JSON)
├── codigos/                        # Scripts Python
│   ├── evaluacion_pipeline.py            # Pipeline completo
│   └── extraccion_tripletas_extranjeras.py
├── informe.tex                     # Informe completo LaTeX
└── requirements.txt                # Dependencias
```

### Composición del Dataset

| Categoría                   | Tripletas      | Entidades | Fuente              |
| --------------------------- | -------------- | --------- | ------------------- |
| **Medicamentos**            | 45.158 (57,7%) | 1.948     | MedlinePlus ES      |
| **Enfermedades**            | 13.391 (17,1%) | 945       | Wikipedia ES        |
| **Artículos Científicos**   | 15.344 (19,6%) | 1.742     | SciELO              |
| **Exámenes**                | 4.315 (5,5%)   | 297       | MedlinePlus ES      |
| **TOTAL Latinoamérica**     | **78.208**     | **4.932** | -                   |
| **TOTAL USA (comparación)** | **76.234**     | **8.809** | MedlinePlus/Wiki EN |

### Ejemplos de Tripletas

```json
{
  "entidad": "Abacavir",
  "relacion": "trata",
  "valor": "infección por el virus de la inmunodeficiencia humana (VIH)"
}
```

**Contenido Regional Específico:**

- `⟨cáncer gástrico, es la, primera causa de muerte por cáncer en Chile⟩`
- `⟨población mapuche, tiene, mayor prevalencia de cáncer de vesícula biliar⟩`
- `⟨población aymara, tiene, bajo acceso a atención médica terciaria⟩`

---

## Resultados de Evaluación

### Resumen Global (131.088 evaluaciones)

| Modelo             | Precisión Global | Medicamentos | Enfermedades | Exámenes | Artículos |
| ------------------ | ---------------- | ------------ | ------------ | -------- | --------- |
| **Qwen 2.5 7B**    | **38,94%**       | 43,77%       | 44,64%       | 41,09%   | 30,14%    |
| **Llama 3.1 8B**   | **18,39%**       | 12,00%       | 30,10%       | 33,98%   | 17,80%    |
| **Mistral 0.3 7B** | **39,92%**       | 40,01%       | 53,94%       | 49,77%   | 31,56%    |

**Significancia estadística:** Todas las diferencias son estadísticamente significativas (p < 0,001)

### Comparación Latinoamérica vs USA

| Modelo             | LatAm  | USA      | Δ           | Interpretación   |
| ------------------ | ------ | -------- | ----------- | ---------------- |
| **Qwen 2.5 7B**    | 38,94% | 35,58%   | -3,36%      | Mejor en LatAm   |
| **Llama 3.1 8B**   | 18,39% | 33,15%   | **+14,76%** | Sesgo anglosajón |
| **Mistral 0.3 7B** | 39,92% | 21,64%\* | -18,28%     | Mejor en LatAm   |

\*Solo medicamentos para Mistral en USA

### Hallazgos Principales

1. **Brecha Regional:** Los artículos científicos latinoamericanos (17-31% precisión) muestran una brecha de ~20 puntos vs conocimiento universal (30-54%), evidenciando subrepresentación sistemática.

2. **Sesgos Divergentes:** Llama 3.1 8B muestra marcado sesgo anglosajón (+14,76 puntos en datos USA), mientras Qwen y Mistral tienen mejor desempeño en contenido latinoamericano.

3. **Sensibilidad Farmacológica:** El conocimiento sobre medicamentos presenta la mayor variación geográfica (brechas hasta -19,80 puntos).

---

## Instalación y Uso

### 1. Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `requests`, `aiohttp`, `python-dotenv`

### 2. Configuración

```bash
export OPENROUTER_API_KEY="tu-api-key"
```

O crear archivo `.env`:

```
OPENROUTER_API_KEY=tu-api-key
```

### 3. Ejecutar Pipeline de Evaluación

```python
python codigos/evaluacion_pipeline.py
```

**El pipeline:**

1. Lee tripletas y genera preguntas automáticamente
2. Consulta modelos (Qwen, Llama, Mistral) vía OpenRouter
3. Evalúa respuestas con GPT-4o-mini
4. Guarda resultados en JSON

**Nota:** Proceso completo toma ~15-20 horas y consume créditos de API.

### 4. Análisis de Resultados

```python
import json

# Cargar evaluaciones
with open('evaluaciones/evaluaciones_drugs_qwen.json', 'r', encoding='utf-8') as f:
    evals = json.load(f)

# Calcular precisión
total = len(evals)
correctas = sum(1 for e in evals if e['evaluation'] == 'correct')
print(f"Precisión: {correctas/total*100:.2f}%")

# Ver errores
errores = [e for e in evals if e['evaluation'] == 'incorrect']
for error in errores[:5]:
    print(f"\nPregunta: {error['question']}")
    print(f"Esperada: {error['expected_answer']}")
    print(f"Obtenida: {error['model_answer']}")
```

---

## Formato de Datos

### Tripletas

```json
[
  {
    "entidad": "Niacina",
    "relacion": "dieta",
    "valor": "seguir una dieta baja en grasa y colesterol"
  }
]
```

### Preguntas

```json
[
  {
    "question": "What conditions or diseases does Abacavir treat?",
    "answer": "infección por el virus de la inmunodeficiencia humana (VIH)"
  }
]
```

### Evaluaciones

```json
[
  {
    "id": 1,
    "question": "What conditions or diseases does Abacavir treat?",
    "expected_answer": "infección por VIH",
    "model_answer": "Abacavir is used to treat HIV infection...",
    "evaluation": "correct",
    "processing_date": "2025-10-30T12:34:56.789Z",
    "evaluation_date": "2025-10-30T13:00:00.000Z"
  }
]
```

### Relaciones Principales

**Medicamentos:** `trata`, `efecto_secundario`, `precaucion`, `contraindicado_en`, `uso`, `se_administra_via`, `almacenamiento`, `marca_comercial`

**Enfermedades:** `presenta_sintoma`, `causada_por`, `tratada_con`, `diagnosticada_mediante`, `asociada_con`, `afecta_organo`, `es_hereditaria`, `puede_causar`

**Exámenes:** `diagnostica`, `mide`, `detecta`, `requiere_muestra_de`, `tiene_riesgo`, `tiene_duracion`, `realizado_por`

---

## Metodología

### Construcción del Dataset

1. **Extracción NLP:** spaCy para identificar relaciones en textos estructurados
2. **Generación LLM:** GPT-4o-mini para extraer valores de las relaciones
3. **Filtrado:** Pipeline de doble validación para artículos científicos
4. **Calidad:** Tasa de validez >98% en medicamentos/enfermedades (revisión manual)

### Evaluación

- **Modelos evaluados:** Qwen 2.5 7B, Llama 3.1 8B, Mistral 0.3 7B
- **Total instancias:** 131.088 preguntas
- **Evaluador:** GPT-4o-mini (evaluación binaria: correcto/incorrecto)
- **Criterios:** Precisión factual, equivalencia semántica, relevancia

---

## Estadísticas Clave

### Distribución de Tripletas por Relación (Top 10)

1. **efecto_secundario**: ~8.500
2. **trata**: ~6.200
3. **presenta_sintoma**: ~5.800
4. **precaucion**: ~4.900
5. **uso**: ~4.200
6. **causada_por**: ~3.100
7. **diagnostica**: ~2.800
8. **contraindicado_en**: ~2.500
9. **tratada_con**: ~2.300
10. **asociada_con**: ~2.100

### Tamaños de Archivos

- **Datasets (tripletas):** ~30 MB total
- **Preguntas:** ~19 MB total
- **Respuestas:** ~180 MB total
- **Evaluaciones:** ~121 MB total
- **Total repositorio:** ~301 MB

### Promedios por Entidad

- **Medicamentos:** 23,2 tripletas/medicamento
- **Enfermedades:** 14,2 tripletas/enfermedad
- **Exámenes:** 14,5 tripletas/examen
- **Artículos:** 8,8 tripletas/artículo

---

## Limitaciones

1. **Dependencia en GPT-4o-mini:** Para generación de valores y evaluación
2. **Tamaño desigual:** Categorías con diferentes números de instancias
3. **Sin validación médica:** No revisado por expertos clínicos
4. **Cobertura geográfica:** Principalmente Chile, limitada representación de otros países latinoamericanos
5. **Idioma mixto:** Preguntas en inglés, respuestas esperadas en español (según fuente original)

---

## Citación

```bibtex
@article{burgos2025grafo,
  title={Grafo de Conocimiento Médico Latinoamericano: Evaluación de Brechas Regionales en Modelos de Lenguaje},
  author={Burgos, Sebastián and Fernández, Gonzalo and Sumonte, Nicolás},
  year={2025},
  institution={Pontificia Universidad Católica de Chile}
}
```

---

## Licencia

Este dataset y código están disponibles para uso académico. Por favor cite este trabajo si lo utiliza en su investigación.

---

## Solución de Problemas

**Error: "OPENROUTER_API_KEY not found"**

- Configure la variable de entorno o cree archivo `.env`

**Rate limit exceeded**

- Los scripts incluyen reintentos automáticos con backoff
- Reduzca `MAX_WORKERS` en los scripts (default: 15)

**Archivos JSON muy grandes**

- Use `jq` para procesamiento eficiente:
  ```bash
  cat evaluaciones_drugs_qwen.json | jq '[.[] | select(.evaluation=="correct")] | length'
  ```

**Timeout en llamadas API**

- Verifique su conexión a internet
- Algunos modelos son más lentos que otros
- El timeout por defecto es 60 segundos

---

## Contacto

Para preguntas o comentarios, abra un issue en: https://github.com/VadokDev/grafo-conocimiento-medico

---

**Última actualización:** Octubre 2025  
**Versión:** 1.0
