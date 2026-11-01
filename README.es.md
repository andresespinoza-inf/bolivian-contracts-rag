# Bolivian Contracts RAG

**Recuperación y verificación sensibles a la estructura sobre contratos bolivianos y normas del Código Civil** — diseñado para que cada afirmación pueda rastrearse hasta la cláusula o el artículo exacto del que proviene, y para que el sistema diga *"no verificado"* en lugar de adivinar.

[English](README.md) · **Español**

El corpus jurídico y los ejemplos están en español; el código, los tests y la documentación están en inglés.

> **Estado: prototipo de investigación funcional.** Las capas de extracción, recuperación, orquestación de verificación y consistencia de partes están implementadas y cubiertas por 33 tests que se ejecutan sin modelos, GPU ni claves de API. La generación de respuestas, la detección de contradicciones y la interfaz se están construyendo localmente y **todavía no están en este repositorio**; consulta [Qué funciona hoy](#qué-funciona-hoy) para ver el límite exacto.

## Por qué esto no es "chat con tu PDF"

Un RAG de tutorial (fragmentos de tamaño fijo, embeddings por defecto, un prompt de una línea) encaja mal en el derecho, por razones propias del dominio:

| Supuesto del RAG genérico | Qué falla en contratos y códigos | Qué hace este proyecto en su lugar |
|---|---|---|
| Dividir el texto cada N tokens | Separa una regla de su excepción dentro de la misma cláusula; rompe las referencias cruzadas | Divide según la estructura propia del documento (cláusula / artículo / considerando), conserva la cláusula completa como padre y subdivide las largas por sus propios incisos numerados |
| La similitud semántica es suficiente | "Artículo 1538" o "cláusula quinta" deben coincidir *exactamente* | Recuperación híbrida: BM25 + vectores fusionados con Reciprocal Rank Fusion y luego reranking |
| Confiar en la cita del LLM | Una cláusula o un artículo inventado es responsabilidad profesional, no una errata | Cada afirmación se contrasta con el fragmento que cita; si no se puede respaldar, se muestra como **no verificada**, nunca se oculta ni se promueve |
| Un solo parser sirve para todo | Los contratos bolivianos no tienen sección de considerandos; los españoles sí | Los patrones por jurisdicción viven en YAML ([`configs/`](contracts_rag/configs)); añadir una jurisdicción es un cambio de datos, no del motor |
| Todo necesita un modelo | Algunas comprobaciones jurídicas son deterministas | Donde la regla es mecánica (¿coinciden los firmantes con las partes nombradas en la cláusula primera?), es una función simple, con la evidencia adjunta |

## Qué funciona hoy

| Componente | Estado | Notas |
|---|---|---|
| Extracción de estructura (`RegexStructureExtractor`) | ✅ implementado y probado | Cláusulas, artículos, considerandos, secciones, subfragmentos, referencias cruzadas y marcas de confianza |
| Configuración por jurisdicción (contrato BO, contrato ES, ley BO) | ✅ implementado y probado | YAML, sin cambios de código para añadir una jurisdicción |
| Comprobación determinista de consistencia de partes | ✅ implementado y probado | Solo formato boliviano; señala, nunca concluye invalidez |
| Recuperación híbrida: BM25 + fusión RRF | ✅ implementado y probado | Se ejecuta en cualquier equipo, sin descargas |
| Vector + rerank, pila clásica (TF-IDF, solapamiento de tokens) | ✅ implementado y probado | Sustituto deliberadamente débil para que el pipeline funcione sin descargas |
| Vector + rerank, pila neuronal (bge-m3, bge-reranker) | 🧪 implementado, **aún sin benchmarks** | `pip install -e ".[models]"` |
| Verificador de dos niveles — orquestación | ✅ implementado y probado con dobles de prueba | El nivel 2 es opcional; sin él, los resultados ambiguos quedan como "no verificados" |
| Verificador nivel 1 — modelo NLI | 🧪 implementado, **aún sin benchmarks** | `cross-encoder/nli-deberta-v3-base` |
| Verificador nivel 2 — LLM | 🧪 solo interfaz y prompt restringido | Ningún LLM está conectado ni evaluado en este repositorio |
| Detección de contradicciones entre documentos | 🚧 en desarrollo local | Reutilizará el verificador |
| Generación de respuestas con estado de verificación por afirmación | 🚧 en desarrollo local | Prevista con un LLM local (Ollama) |
| Interfaz con resaltado de fuentes | 🚧 planificada | |
| Conjunto de evaluación (recall@k, precisión de citas) | 🚧 planificado | Actualmente **no hay métricas de recuperación**; consulta Limitaciones |

## Inicio rápido

```bash
git clone https://github.com/andresespinoza-inf/bolivian-contracts-rag.git
cd bolivian-contracts-rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                               # 33 tests, ~1 s, sin modelos

python examples/01_extract_structure.py
python examples/02_party_consistency.py
python examples/03_hybrid_retrieval.py

# Opcional — modelos neuronales reales (descarga ~4 GB desde Hugging Face la primera vez)
pip install -e ".[models]"
python examples/03_hybrid_retrieval.py --real-models
python examples/04_verify_claims.py
```

## Ejemplo: una alerta de formación del consentimiento, sin ningún modelo

En la doctrina boliviana, los cuatro requisitos de formación del contrato son consentimiento, objeto, causa y forma. El consentimiento aparece dos veces en un contrato —las partes identificadas en la cláusula primera y los firmantes al final— y deben coincidir. `examples/02_party_consistency.py` comprueba exactamente eso sobre el contrato sintético incluido, donde el bloque de firmas nombra deliberadamente a otras personas:

```text
status: INCONSISTENT
parties clause : ['ROBERTO CALLE MAMANI', 'SILVIA CALLE MAMANI']
closing clause : ['ELIAS FLORES CONDORI', 'TERESA ROJAS LIMACHI']

Names identified in the parties clause do not match the names in the closing/signature
clause. Requires lawyer review (possible consent-formation issue); this is a flag, not
a finding of invalidity.
evidence chunk ids (for highlighting): ['5f26…', 'ca86…']      # los ids cambian en cada ejecución
```

El informe incluye los ids de ambos fragmentos de origen para que una interfaz pueda resaltarlos en paralelo. El encuadre jurídico (error, dolo, nulidad vs. anulabilidad) vive en el extracto del Código Civil incluido (arts. 473–482, 549–559), de modo que el sistema puede *recuperar* la norma en lugar de afirmarla de memoria.

## Ejemplo: recuperación híbrida sobre el corpus incluido

Salida de `examples/03_hybrid_retrieval.py` con la **pila clásica** (sin descargas). Las etiquetas están truncadas y las flechas son anotaciones mías; el ranking en sí no está editado, incluidas sus debilidades:

```text
Q: ¿Qué pasa si el contrato de compraventa no consta en documento público?
  #1  CONTRATO DE COMPRAVENTA DE INMUEBLE Conste por el presente doc…   (el preámbulo de "documento privado")
  #2  SEXTA. - ESCRITURA PÚBLICA. La escritura pública de compravent…
  #3  Artículo 491°.- (Contratos y actos que deben hacerse por docum…
  #4  Artículo 569°.- (Cláusula resolutoria) …                           ← acierto débil

Q: nulidad del contrato por vicios del consentimiento
  #1  Artículo 556°.- (Prescriptibilidad de la acción de anulación) …
  #2  Artículo 550°.- (Nulidad parcial del contrato) …
  #3  Artículo 549°.- (Casos de nulidad del contrato) …
  #4  Artículo 633°.- (Perecimiento de la cosa) …                        ← acierto irrelevante
```

Las normas correctas aparecen, pero la cola es ruidosa. Eso es lo que produce una pila léxica/TF-IDF; si la pila neuronal lo mejora es algo que **todavía no he medido** y no quiero afirmar.

## Decisiones de diseño

1. **Fragmentar según la estructura jurídica, indexar solo las hojas.** La cláusula o el artículo es la unidad atómica. Si supera el umbral de tokens, se divide por sus propios sub-marcadores (`2.1.`, `a)`, `I.`), con una ventana de tokens simple solo como último recurso, registrada como tal. El índice contiene únicamente las unidades hoja, nunca un padre *y* sus hijos, lo que duplicaría el mismo texto.
2. **Nunca descartar texto en silencio.** Todo lo que no coincide con un marcador conocido se conserva como `fragmento_sin_clasificar` con `confidence=baja`. No es cosmético: la línea inicial del contrato boliviano incluido ("documento privado, que podrá ser elevado a público…") está *antes* de la cláusula primera y es precisamente la evidencia de un riesgo de forma/registro.
3. **La configuración es datos.** Los patrones de marcadores por jurisdicción y tipo de documento viven en YAML. La plantilla española tiene considerandos numerados (`EXPONEN I–III`) que la boliviana no tiene; al motor le da igual.
4. **Verificación en dos niveles, con el segundo opcional.** Un clasificador NLI dedicado contrasta cada afirmación con el fragmento que cita (barato, determinista y de un tipo de modelo distinto al que generó la afirmación). Solo los casos ambiguos pasan a un LLM, con una respuesta restringida de tres opciones (*supports / contradicts / not applicable*), porque un sí/no no distingue "no lo menciona" de "lo contradice". Sin LLM configurado, los casos ambiguos se informan como *no verificados*, y una etiqueta NLI de baja confianza nunca se promueve a verificada.
5. **Fusión de rangos, no de puntuaciones.** Las puntuaciones BM25 no tienen cota; la similitud coseno sí. Las sumas ponderadas son frágiles entre corpus; RRF usa solo los rangos.
6. **Determinista donde la ley es determinista.** La consistencia entre firmas y partes es una función, no un prompt.

## Notas de ingeniería: lo que me enseñaron los PDFs reales

El parser se desarrolló sobre contratos reales mantenidos en privado (no redistribuidos) y luego se volvió a verificar con los fixtures sintéticos de este repositorio, que reproducen las mismas particularidades.

- **Los encabezados se parten.** La extracción de PDF divide `SEXTA (OBLIGACIONES / DE LAS PARTES).-` en dos líneas; los números romanos españoles llegan solos en su propia línea. Por eso los marcadores no pueden exigir su puntuación de cierre en la misma línea.
- **La coincidencia sin distinguir mayúsculas crea fragmentos falsos.** Una línea partida que empieza con la palabra *segunda* en "…la cláusula / segunda…" no debe abrir una nueva cláusula; la coincidencia de marcadores distingue mayúsculas a propósito.
- **La pérdida silenciosa es fácil de escribir.** Dos bugs de este tipo aparecieron al construir casos de prueba con documentos reales: un preámbulo descartado y —al dividir una cláusula larga por sus incisos numerados— la frase introductoria antes del primer inciso, que se perdía (en el fixture español, la frase que fija el precio). El segundo tiene un test de regresión que reconstruye cada cláusula padre a partir de sus subfragmentos y falla si falta una sola palabra; comprobé que el test falla de verdad cuando se elimina la corrección.
- **El sesgo de frecuencia de términos es real.** En los documentos reales, una cláusula larga de calendario de pagos superaba en ranking a otra más corta y directamente relevante con BM25/TF-IDF. Es el tipo de caso que motiva la recuperación semántica; todavía no he cuantificado la mejora.

## Limitaciones (leer antes de juzgar el proyecto)

- **Corpus diminuto y fixtures sintéticos.** Dos contratos de compraventa sintéticos (Bolivia y España) y un extracto del Código Civil boliviano compilado a mano (122 artículos, vigente desde el 2 de abril de 1976 según su art. 1570; **verificar contra el texto oficial antes de confiar en él**). El fixture boliviano deriva de un contrato real con todos los identificadores reemplazados; el español es texto original modelado sobre la estructura de una plantilla notarial.
- **Todavía no hay evaluación.** No existe un conjunto de preguntas etiquetadas, por lo que no hay cifras de recall@k ni de precisión de citas. Cualquier afirmación sobre la calidad de la recuperación es cualitativa hasta que exista.
- **Componentes neuronales sin benchmarks.** Las rutas bge-m3 / bge-reranker / NLI están implementadas, pero no he publicado resultados.
- **La consistencia de partes es una heurística.** Lee secuencias de nombres en mayúsculas, está calibrada solo con contratos bolivianos y devuelve `not_evaluable` en otros casos en lugar de adivinar.
- **La extracción por regex tiene límites.** Maneja los formatos configurados en YAML. Un layout nuevo necesita una configuración nueva, y los documentos escaneados necesitan OCR aguas arriba.
- **No es asesoría legal.** Es un prototipo de investigación. Sus alertas son indicios para la revisión de un abogado, no conclusiones.

## Roadmap (en desarrollo local)

- Detección de contradicciones entre documentos, reutilizando el verificador sobre fragmentos agrupados por tema
- Generación de respuestas donde cada frase lleve su fuente y su estado de verificación, con un LLM local
- Una interfaz mínima que resalte la cláusula de origen junto a la respuesta
- Un conjunto de evaluación pequeño etiquetado a mano y métricas de recuperación y citas
- Una segunda jurisdicción completamente por configuración (España está conectada solo para contratos)

## Estructura del repositorio

```text
contracts_rag/
  core/          modelo de datos (Document, Section, Chunk, SubChunk) e interfaces
  extractors/    extractor de estructura por regex
  configs/       patrones por jurisdicción (YAML)
  indexing/      índices BM25, TF-IDF y sentence-transformers
  retrieval/     fusión RRF, recuperador híbrido y rerankers
  verifiers/     nivel NLI, nivel LLM (cliente inyectable) y orquestador de dos niveles
  analysis/      comprobación determinista de consistencia de partes
data/            contratos sintéticos + extracto del Código Civil (ver data/README.md)
examples/        demos ejecutables
tests/           33 tests
docs/            [registro de decisiones de arquitectura](docs/architecture-decisions.md)
```

## Acerca de

Creado por un abogado boliviano (titulado en 2022; maestría en derecho civil y de familia en curso) que también estudia informática, con enfoque en IA y ciencia de datos. El conocimiento del dominio es el punto: el diseño sigue cómo se forman y se impugnan realmente los contratos en el derecho civil boliviano, en lugar de tratar el texto jurídico como prosa genérica.

## Licencia

MIT — ver [LICENSE](LICENSE). El extracto del Código Civil incluido es texto legal público; los fixtures no contienen datos personales reales.
