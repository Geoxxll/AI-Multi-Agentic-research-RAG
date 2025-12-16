import re
from pypdf import PdfReader
from transformers import pipeline
import pprint

def fix_broken_words(text):
    # Fix: "I NTRODUCTION" → "INTRODUCTION"
    # Fix: "R ELATED WORK" → "RELATED WORK"
    text = re.sub(r"([A-Z])\s+([A-Z])", r"\1\2", text)
    return text

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_entities(words):
    cleaned = set()
    for w in words:
        w2 = w.replace("-", "").replace(" ", "")
        if len(w2) < 3:
            continue
        cleaned.add(w2)
    return sorted(cleaned)

def extract_title(text):
    """
    Extract academic paper title (robust for ICLR / NeurIPS / arXiv / CVPR).
    """

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln][:60]  # only look at first 60 lines

    cleaned = []
    for ln in lines:
        # skip common non-title lines
        low = ln.lower()
        if any(x in low for x in [
            "under review", "iclr", "arxiv", "conference", "workshop",
            "anonymous", "authors", "@", "university", "institute", "lab"
        ]):
            continue

        # skip lines that end with . (not a title)
        if ln.endswith("."):
            continue

        # fix broken capital spacing (Z E R O -> ZERO)
        ln = re.sub(r"([A-Z])\s+([A-Z])", r"\1\2", ln)

        # heuristic: title length ~10–150 chars
        if 10 <= len(ln) <= 150:
            cleaned.append(ln)

    # If no candidate found
    if not cleaned:
        return None

    # If title is multiline, collect consecutive lines
    final = [cleaned[0]]
    i = lines.index(cleaned[0]) + 1
    while i < len(lines):
        ln = lines[i].strip()
        if ln and not ln.endswith("."):
            final.append(ln)
            i += 1
        else:
            break

    title = " ".join(final)
    title = re.sub(r"\s+", " ", title)
    return title.strip()



def extract_abstract(text):
    patterns = [
        # Abstract 到 Introduction
        r"(?i)abstract[:\s]*(.*?)(?=\n\s*(introduction|1\.|I\.)\b)",
        
        # Abstract 到下一全大写标题（适配 IEEE）
        r"(?i)abstract[:\s]*(.*?)(?=\n[A-Z][A-Z\s]{3,}\n)",

        # Abstract 行后面紧接正文（适配一些 Latex）
        r"(?i)^abstract\s*\n(.*?)(?=\n\s*[A-Z][a-z]+)",
        
        # Abstract: 第一段
        r"(?i)abstract[:\s]*(.*?)(\n\n|\r\n\r\n)"
    ]

    for p in patterns:
        m = re.search(p, text, re.DOTALL)
        if m:
            return m.group(1).strip()

    return None


def extract_sections(text):
    text = fix_broken_words(text)

    # Example: "3EXPERIMENTSETTINGS" -> "3 EXPERIMENT SETTINGS"
    text = re.sub(r"(\d)([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})\s*([A-Z]{2,})", r"\1 \2", text)

    pattern = r"""
        (?m) ^
        \s*
        (?P<num>\d{1,2}(\.\d{1,2})*)
        \s+
        (?P<title>[A-Z][A-Z\s]{3,})
    """

    sections = []
    for m in re.finditer(pattern, text, flags=re.VERBOSE):
        title = m.group("title").strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) <= 50:   # avoid noise
            sections.append(title)

    # unique
    out, seen = [], set()
    for s in sections:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out




def extract_entities(text):
    # Pre-lower for matching
    lower = text.lower()

    # ====== 1. DATASETS ======
    dataset_patterns = [
        r"imagenet", r"cifar[- ]?10", r"cifar[- ]?100", r"mnist",
        r"coco", r"squad", r"wikitext[- ]?2", r"imdb",
        r"quora", r"reddit", r"libri\s*s?peech?", r"ptb",
        r"sst[- ]?2", r"cityscapes", r"pascal voc", r"kitti",
        r"celeba", r"yelp", r"enron"
    ]

    datasets = set()
    for pat in dataset_patterns:
        for m in re.findall(pat, lower):
            datasets.add(m)

    # ====== 2. METRICS ======
    metric_patterns = [
        r"accuracy", r"f1[- ]?score", r"precision", r"recall",
        r"bleu", r"meteor", r"rouge", r"auc", r"roc",
        r"psnr", r"ssim", r"rmse", r"mae"
    ]

    metrics = set()
    for pat in metric_patterns:
        for m in re.findall(pat, lower):
            metrics.add(m)

    # ====== 3. METHODS ======
    method_patterns = [
        r"transformer", r"bert", r"gpt[- ]?\d*", r"llama",
        r"vit", r"cnn", r"rnn", r"lstm", r"gru",
        r"unet", r"resnet", r"densenet", r"gan", r"vae",
        r"diffusion", r"graph neural network", r"gnn"
    ]

    methods = set()
    for pat in method_patterns:
        for m in re.findall(pat, lower):
            methods.add(m)

    return {
        "methods": clean_entities(methods),
        "datasets": sorted(list(datasets)),
        "metrics": sorted(list(metrics))
    }



def build_paper_signature(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text = fix_broken_words(text)   # <-- add this

    title = extract_title(text)
    abstract = extract_abstract(text)
    sections = extract_sections(text)
    entities = extract_entities(text)

    # ---- Derive Topic ----
    if abstract:
        # clean abstract
        abs_clean = abstract.strip().replace("\n", " ")

        # common stop openers
        common_openers = [
            "in recent years",
            "recently",
            "with the development of",
            "with the advance of",
        ]

        # lower for check
        abs_lower = abs_clean.lower()

        topic_sentence = abs_clean.split(".")[0]

        # If the first sentence is boilerplate, pick next
        if any(topic_sentence.lower().startswith(op) for op in common_openers):
            parts = abs_clean.split(".")
            if len(parts) > 1:
                topic_sentence = parts[1]

        # final cleanup
        topic = topic_sentence.strip()
    else:
        topic = title  # fallback

    signature = {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "entities": entities,
        "topic": topic
    }

    return signature


paper_signature = build_paper_signature("/Users/george/ai-projects/MultiAgenticRAG_Rep/papers/2310.08560v2.pdf")
