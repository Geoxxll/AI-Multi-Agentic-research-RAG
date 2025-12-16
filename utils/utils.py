import yaml
import uuid
from langchain_core.documents import Document
from typing import List, Dict, TypedDict
def load_config(file_path="./config.yaml"):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def reduce_docs(a: list, b: list) -> list:
    return a + b

def new_uuid():
    return str(uuid.uuid4())

async def align_evidence_to_steps(
    model,
    steps: list[str],
    evidence: list[str],
):
    """
    Returns:
    {
      step_idx: [evidence_idx, ...]
    }
    """
    prompt = f"""
You are an evidence alignment assistant.

Given:
- A list of research steps
- A list of evidence items

For EACH research step, select ONLY the evidence items
that directly contain factual information relevant to that step.

Rules:
- If no evidence applies, return an empty list for that step.
- Do NOT infer or guess.
- Selection only, no explanation.

Return JSON in the following format:
{{
  "alignment": {{
    "0": [1, 3],
    "1": [],
    ...
  }}
}}

Research Steps:
{steps}

Evidence:
{[f"Evidence {i+1}: {e}" for i, e in enumerate(evidence)]}
"""
    class Alignment(TypedDict):
        alignment: Dict[str, list]
    response = await model.with_structured_output(Alignment).ainvoke(
        [{"role": "system", "content": prompt}]
    )
    return response["alignment"]


async def write_step_from_evidence(
    model,
    step: str,
    selected_evidence: list[str],
):
    if not selected_evidence:
        return "The provided evidence does not contain information about this aspect."

    prompt = f"""
You are a grounded writing assistant.

Task:
Rewrite the following evidence into a coherent paragraph
that answers the research step.

STRICT RULES:
- Use ONLY the provided evidence.
- Do NOT add new facts.
- Do NOT generalize.
- Do NOT introduce generic statements.
- Every sentence must be traceable to the evidence.

Research Step:
{step}

Evidence:
{selected_evidence}

==================
OUTPUT FORMAT
==================
OUTPUT JSON ONLY:
    {{
    "paragraph": "..."
    }}

"""
    class StepParagraph(TypedDict):
        paragraph: str

    response = await model.with_structured_output(StepParagraph).ainvoke(
        [{"role": "system", "content": prompt}]
    )
    return response

config = load_config()