"""
Prompt templates for the clinical note generator.
All templates use f-string-compatible placeholders wrapped in { }.
"""

SYSTEM_PROMPT = """You are an expert clinical AI assistant trained on MIMIC-III clinical notes.
You generate structured SOAP (Subjective, Objective, Assessment, Plan) notes for physicians.
Always be medically precise, cite findings explicitly, and flag uncertainties.
Do NOT fabricate lab values or imaging results not present in the input.
Format your output strictly as the structured JSON schema requested."""

SOAP_GENERATION_TEMPLATE = """## CLINICAL NOTE GENERATION TASK

### DOCTOR'S TRANSCRIPTION (Subjective)
{transcription}

### RADIOLOGY IMAGE FINDINGS (Objective)
Imaging modality: {imaging_modality}
Top detected pathologies (CheXNet confidence scores):
{pathology_findings}

### RETRIEVED MEDICAL LITERATURE (Context)
{rag_context}

---

### INSTRUCTIONS
Based on the transcription, imaging findings, and literature context above, generate a complete clinical SOAP note.
Return ONLY a valid JSON object with this exact structure:

{{
  "soap_note": {{
    "subjective": "<patient history, chief complaint, and symptoms from transcription>",
    "objective": "<vital signs, imaging findings, and observable data>",
    "assessment": "<differential diagnoses ranked by likelihood with reasoning>",
    "plan": "<treatment plan, medications, follow-up, referrals>"
  }},
  "primary_diagnosis": "<most likely diagnosis>",
  "differential_diagnoses": ["<dx1>", "<dx2>", "<dx3>"],
  "image_findings_summary": "<one paragraph summarizing imaging>",
  "recommended_medications": ["<med1>", "<med2>"],
  "follow_up": "<follow-up instructions>",
  "confidence": "<high|medium|low>",
  "literature_references": ["<ref1>", "<ref2>", "<ref3>"]
}}"""

RAG_CONTEXT_TEMPLATE = """[{index}] {title}
Source: PubMed PMID {pmid}
Abstract: {abstract}
---"""

DRUG_INTERACTION_TEMPLATE = """You are a clinical pharmacist AI. Check the following medication list for interactions.

Medications: {medications}

Return a JSON array of interaction warnings:
[
  {{
    "drug_a": "<drug1>",
    "drug_b": "<drug2>",
    "severity": "<major|moderate|minor>",
    "description": "<clinical description of interaction>",
    "recommendation": "<clinical action>"
  }}
]
If no interactions are found, return an empty array: []"""

NER_HIGHLIGHT_TEMPLATE = """Extract all medical named entities from the following clinical text.

Text: {text}

Return JSON with these entity categories:
{{
  "symptoms": ["<symptom1>", ...],
  "diagnoses": ["<dx1>", ...],
  "medications": ["<med1>", ...],
  "dosages": ["<dosage1>", ...],
  "procedures": ["<proc1>", ...],
  "anatomy": ["<body_part1>", ...]
}}"""
