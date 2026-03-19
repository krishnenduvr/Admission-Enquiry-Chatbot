import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

CHUNKS_PATH = "pdf_chunks.json"
EXTRACTED_TEXT_PATH = "Nmcc_english_relevant_extracted_from_py1.txt"
INTENTS_PATHS = [
    r"d:\Nesamony Memorial Christian College\intents.json",
    "intents.json",
]

MAX_SECTION_CHARS = 200000
UNKNOWN_REPLY = (
    "Sorry, I couldn't find that in this document. "
    "Please ask about NMCC."
)
GREETING_REPLY = "Hello! Ask me anything about the NMCC handbook."
BYE_REPLY = "Goodbye! If you need NMCC details again, just ask."


def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_output_text(s: str) -> str:
    s = normalize_mojibake(s)
    s = re.sub(r"(?im)^\s*---\s*Page\s+\d+\s*---\s*$", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return normalize_text(s)


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", s.lower())


def load_intents() -> List[Dict[str, object]]:
    for intents_path in INTENTS_PATHS:
        try:
            if not os.path.exists(intents_path):
                continue
            with open(intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        intents = data.get("intents", [])
        if isinstance(intents, list):
            return [item for item in intents if isinstance(item, dict)]

    return []


STOP_TOKENS = {
    "what", "are", "is", "the", "a", "an", "of", "about", "tell", "me", "explain", "give", "details",
    "for", "on", "in", "to", "and", "please", "topic",
    "do", "you", "have", "has", "had", "any", "there", "your", "our", "can", "could", "would",
    "show", "provide", "share", "want", "need", "if", "yes", "no",
    "college", "nmcc", "institution",
}

ALLOWED_ACRONYMS = {
    "nmcc",
    "ug",
    "pg",
    "bsc",
    "msc",
    "bcom",
    "mcom",
    "bca",
    "mca",
    "bba",
    "mba",
    "phd",
    "ncc",
    "nss",
}


def normalize_mojibake(s: str) -> str:
    replacements = {
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u20ac\u00a6": "...",
        "\u00c2": "",
        "\ufffd": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00c5\u00b8": "-",
        "Å¸": "-",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


def normalize_heading_key(s: str) -> str:
    s = normalize_mojibake(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    fixes = {
        "universtiy": "university",
        "students council": "students council",
        "botony": "botany",
        "committe": "committee",
        "commitee": "committee",
        "graduation committe": "graduation committee",
        "gradution committee": "graduation committee",
    }
    for bad, good in fixes.items():
        s = s.replace(bad, good)
    return s


def content_tokens(s: str) -> List[str]:
    return [t for t in tokenize(s) if t not in STOP_TOKENS]


def keyword_in_text(text: str, keyword: str) -> bool:
    text_l = text.lower()
    key = keyword.lower().strip()
    if not key:
        return False
    if " " in key:
        return key in text_l
    return re.search(rf"\b{re.escape(key)}\b", text_l) is not None


def load_source_text(chunks: List[str]) -> str:
    try:
        with open(EXTRACTED_TEXT_PATH, "r", encoding="utf-8") as f:
            text = normalize_mojibake(f.read())
            # If key headings are missing (e.g., Rules/University), fall back to chunks.
            if (
                ("RULES AND REGULATIONS" not in text.upper() and "RULES & REGULATIONS" not in text.upper())
                or ("OUR UNIVERSITY" not in text.upper() and "OUR UNIVERSTIY" not in text.upper())
            ):
                return normalize_mojibake("\n".join(chunks))
            return text
    except OSError:
        return normalize_mojibake("\n".join(chunks))


def load_extracted_text_fallback() -> str:
    try:
        with open(EXTRACTED_TEXT_PATH, "r", encoding="utf-8") as f:
            return normalize_mojibake(f.read())
    except OSError:
        return ""


def is_main_heading(line: str) -> bool:
    line = normalize_mojibake(line)
    # Ignore table-of-contents style lines.
    if "..." in line:
        return False
    # Numbered chapter-style headings.
    if re.match(r"^\d+\.\s+[A-Z][A-Za-z0-9 '&()/.-]{4,}$", line):
        body = re.sub(r"^\d+\.\s+", "", line).strip()
        if len(body) > 70:
            return False
        if body.endswith("."):
            return False
        if re.match(r"^(B\.A|B\.Sc|B\.Com|BBA|BCA|M\.A|M\.Sc|M\.Com|MBA|MCA|Ph\.D)\b", body, flags=re.I):
            return False
        if re.match(r"^[A-Z]\.(?:[A-Z]\.)+", body):
            return False
        if len(body.split()) < 3:
            return False
        words = re.findall(r"[A-Za-z']+", body)
        if not words:
            return False
        lower_words = [w for w in words if w.islower()]
        if len(lower_words) > max(2, len(words) // 3):
            return False
        return True
    # Handle all-caps style headings.
    if re.match(r"^[A-Z][A-Z0-9 '&()/.-]{6,}$", line) and len(line.split()) <= 14:
        return True
    return False


def trim_section_by_heading(heading: str, block: str) -> str:
    h = heading.lower()
    lines = block.splitlines()
    text = "\n".join(lines)

    if "campus facilities and students" in h:
        cut_markers = [
            "a) General Discipline",
            "8. RULES AND REGULATIONS",
            "9. STUDENTS’ COUNCIL",
            "9. STUDENTS' COUNCIL",
        ]
        cut_at = len(text)
        for marker in cut_markers:
            pos = text.find(marker)
            if pos >= 0:
                cut_at = min(cut_at, pos)
        text = text[:cut_at]

    return text.strip()


def extract_rules_block(source_text: str) -> str:
    text = normalize_mojibake(source_text)

    start_markers = [
        "8. RULES AND REGULATIONS",
        "a) General Discipline",
    ]
    end_markers = [
        "9. STUDENTS' COUNCIL",
        "9. STUDENTS’ COUNCIL",
        "10. OUR UNIVERSITY",
        "10. Our University",
    ]

    start = -1
    for marker in start_markers:
        pos = text.find(marker)
        if pos >= 0:
            start = pos
            break
    if start < 0:
        return ""

    end = len(text)
    for marker in end_markers:
        pos = text.find(marker, start + 1)
        if pos >= 0:
            end = min(end, pos)

    block = text[start:end].strip()
    return block[:MAX_SECTION_CHARS]


def extract_block_between_markers(
    source_text: str, start_markers: List[str], end_markers: List[str]
) -> str:
    text = normalize_mojibake(source_text)
    start = -1
    for marker in start_markers:
        pos = text.find(marker)
        if pos >= 0:
            start = pos
            break
    if start < 0:
        return ""

    end = len(text)
    for marker in end_markers:
        pos = text.find(marker, start + 1)
        if pos >= 0:
            end = min(end, pos)

    block = text[start:end]
    lines = [normalize_text(x) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    return "\n".join(lines).strip()


def extract_block_between_heading_lines(
    source_text: str, start_patterns: List[str], end_patterns: List[str]
) -> str:
    text = normalize_mojibake(source_text)
    # Find first start match on a full line (avoid TOC dotted lines).
    start = -1
    for pat in start_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            start = m.start()
            break
    if start < 0:
        return ""
    end = len(text)
    if end_patterns:
        for pat in end_patterns:
            m = re.search(pat, text[start + 1 :], flags=re.IGNORECASE | re.MULTILINE)
            if m:
                end = min(end, start + 1 + m.start())
    block = text[start:end]
    lines = [normalize_text(x) for x in block.splitlines()]
    lines = [x for x in lines if x and not is_toc_line(x) and not x.startswith("--- Page")]
    return "\n".join(lines).strip()


def extract_pledge_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["THE PLEDGE"],
        end_markers=["NATIONAL ANTHEM", "CONTENTS", "1. BRIEF HISTORY OF THE COLLEGE"],
    )


def extract_national_anthem_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["NATIONAL ANTHEM"],
        end_markers=["CONTENTS", "1. BRIEF HISTORY OF THE COLLEGE"],
    )


def extract_college_song_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["COLLEGE SONG"],
        end_markers=["THE PLEDGE", "NATIONAL ANTHEM", "CONTENTS"],
    )


def extract_lord_prayer_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["THE LORD'S PRAYER", "THE LORDâ€™S PRAYER"],
        end_markers=["COLLEGE SONG", "THE PLEDGE", "NATIONAL ANTHEM", "CONTENTS"],
    )


def extract_phd_guides_block(source_text: str) -> str:
    text = normalize_mojibake(source_text)
    end_marker = "6. FEE, CONCESSIONS & SCHOLARSHIPS"
    end = text.find(end_marker)
    if end < 0:
        end = len(text)

    start = text.find("Tamil 42. Dr.")
    if start < 0:
        start = text.find("1. Dr. A. Sajan 43. Dr.")
    if start < 0:
        return ""

    raw = text[start:end]
    raw = re.sub(r"---\s*Page\s*\d+\s*---", "\n", raw, flags=re.I)
    raw = re.sub(r"[ \t]+", " ", raw)
    departments = [
        "Management Studies",
        "Computer Science",
        "Mathematics",
        "Economics",
        "Chemistry",
        "Zoology",
        "Physics",
        "Botany",
        "History",
        "English",
        "Tamil",
        "Commerce",
    ]
    for dept in sorted(departments, key=len, reverse=True):
        raw = re.sub(rf"\b{re.escape(dept)}\b", " ", raw, flags=re.I)

    raw = re.sub(r"\s+", " ", raw).strip()

    guides: Dict[int, str] = {}
    for m in re.finditer(r"(\d{1,2})\.?\s*(Dr\.?.*?)(?=(?:\s+\d{1,2}\.?\s*Dr\.?)|$)", raw, flags=re.I):
        n = int(m.group(1))
        if not (1 <= n <= 120):
            continue
        name = normalize_text(m.group(2))
        if not name or n in guides:
            continue
        guides[n] = f"{n}. {name}"

    if not guides:
        return ""

    ordered = [guides[n] for n in sorted(guides)]
    return "\n".join(["5. Ph.D. GUIDES", *ordered]).strip()


def extract_toc_headings(source_text: str) -> List[str]:
    headings: List[str] = []
    for raw in source_text.splitlines():
        line = normalize_text(normalize_mojibake(raw))
        m = re.match(r"^\s*\d+\.\s+(.+?)\s+\.{2,}\s*\d+\s*$", line)
        if not m:
            continue
        title = m.group(1).strip(" .")
        if title and title.lower() != "contents":
            headings.append(title)
    return headings


def parse_main_heading_line(line: str) -> Optional[Tuple[int, str]]:
    m = re.match(r"^\s*(\d{1,2})\.\s+(.+?)\s*$", normalize_mojibake(line))
    if not m:
        return None
    idx = int(m.group(1))
    title = normalize_text(m.group(2)).strip(" .")
    if not title:
        return None
    return idx, title


def is_toc_line(line: str) -> bool:
    line = normalize_mojibake(line)
    if re.search(r"\.{2,}\s*\d+\s*$", line):
        return True
    if "…" in line and re.search(r"\d+\s*$", line):
        return True
    return False


def is_upper_heading(line: str) -> bool:
    line = normalize_mojibake(line)
    if re.match(r"^[A-Z][A-Z0-9 '&()/.-]{3,}$", line) and len(line.split()) <= 10:
        return True
    return False


def is_lettered_heading(line: str) -> bool:
    if re.search(r"\bRs\.?\b", line, flags=re.I):
        return False
    if re.search(r"\d", line):
        return False
    if len(line) > 90:
        return False
    if len(line.split()) > 12:
        return False
    return bool(re.match(r"^[A-Ha-h]\)\s+[A-Za-z][A-Za-z0-9 '&()/.\-]{2,}$", line))


def is_numbered_short_heading(line: str) -> bool:
    m = re.match(r"^(\d{1,3})\.\s+([A-Za-z][A-Za-z0-9 '&()/.\-]{2,})$", line)
    if not m:
        return False
    cand = normalize_text(m.group(2))
    cand_l = cand.lower()
    if len(cand) > 45:
        return False
    if len(cand.split()) > 10:
        return False
    if cand.endswith("."):
        return False
    if cand_l.startswith(("it ", "all ", "students ", "the ")):
        return False
    if re.search(r"\bdr\.?\b", cand_l):
        return False
    return True


def is_any_heading_line(line: str) -> bool:
    if is_toc_line(line):
        return False
    return (
        is_main_heading(line)
        or is_upper_heading(line)
        or is_lettered_heading(line)
        or is_numbered_short_heading(line)
    )


def is_inline_heading_candidate(line: str) -> bool:
    l = normalize_mojibake(line).strip()
    if not l:
        return False
    if is_any_heading_line(l):
        return True
    # Allow Article-style headings.
    if re.match(r"^article[-\s]*[ivx]+", l, flags=re.I):
        return True
    if re.match(r"^article[-\s]*[ivx]+\s*[:\-]", l, flags=re.I):
        return True
    # Short title-case headings without digits.
    if not re.search(r"\d", l) and len(l.split()) <= 6:
        words = re.findall(r"[A-Za-z]+", l)
        if words and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1):
            return True
    return False


def is_heading_only_block(text: str) -> bool:
    lines = [normalize_text(normalize_mojibake(x)) for x in text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    return len(lines) == 1 and is_any_heading_line(lines[0])


def strip_leading_heading_line(block: str) -> str:
    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if len(lines) >= 2 and is_any_heading_line(lines[0]):
        return "\n".join(lines[1:]).strip()
    return "\n".join(lines).strip()


def find_best_intent_match(query: str, intents: List[Dict[str, object]]) -> Tuple[str, float]:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm or not intents:
        return "", 0.0

    q_tokens = set(content_tokens(q_norm))
    best_score = 0.0
    best_response = ""

    for intent in intents:
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        if not isinstance(patterns, list) or not isinstance(responses, list) or not responses:
            continue

        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            p_norm = normalize_heading_key(normalize_mojibake(pattern))
            if not p_norm:
                continue
            p_tokens = set(content_tokens(p_norm))

            score = 0.0
            if q_norm == p_norm:
                score = 3.0
            elif len(p_norm) >= 4 and (q_norm in p_norm or p_norm in q_norm):
                score = 2.0

            if q_tokens and p_tokens:
                overlap = len(q_tokens & p_tokens) / max(1, len(q_tokens))
                score += overlap
                if q_tokens == p_tokens:
                    score += 0.3
                if len(p_tokens) == 1 and len(next(iter(p_tokens), "")) <= 3 and not (p_tokens & q_tokens):
                    score = 0.0

            sim = SequenceMatcher(None, q_norm, p_norm).ratio()
            score += 0.35 * sim

            if score > best_score:
                first_response = next((r for r in responses if isinstance(r, str) and r.strip()), "")
                if first_response:
                    best_score = score
                    best_response = normalize_mojibake(first_response).strip()

    if best_score < 1.15:
        return "", 0.0
    return best_response, best_score


def match_intent_response(query: str, intents: List[Dict[str, object]]) -> str:
    return find_best_intent_match(query, intents)[0]


def build_heading_index_from_text(source_text: str) -> List[Tuple[str, str]]:
    raw_lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in raw_lines if x and not x.startswith("--- Page")]

    heads: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if is_toc_line(line):
            continue
        if is_any_heading_line(line):
            heads.append((i, line))

    if not heads:
        return []

    index: List[Tuple[str, str]] = []
    seen = set()
    for idx, (start_i, heading) in enumerate(heads):
        end_i = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        block_lines = lines[start_i:end_i]
        block = "\n".join(block_lines).strip()
        key = normalize_heading_key(heading)
        if not key or key in seen or not block:
            continue
        seen.add(key)
        index.append((key, block))
    return index


def auto_heading_match(query: str, heading_index: List[Tuple[str, str]]) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm or len(q_norm) < 4:
        return ""
    q_tokens = set(content_tokens(q_norm))

    best_score = 0.0
    best_block = ""
    best_key_len = 0
    for key, block in heading_index:
        score = 0.0
        if key == q_norm:
            score = 3.0
        elif key in q_norm or q_norm in key:
            score = 2.0

        if q_tokens:
            k_tokens = set(content_tokens(key))
            if k_tokens:
                overlap = len(q_tokens & k_tokens) / max(1, len(q_tokens))
                score += overlap
                if q_tokens.issubset(k_tokens):
                    score += 0.25

        if score > best_score or (score == best_score and len(key) > best_key_len):
            best_score = score
            best_block = block
            best_key_len = len(key)

    if best_score < 0.45:
        return ""
    return best_block


def exact_heading_match(query: str, heading_index: List[Tuple[str, str]]) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm or len(q_norm) < 4:
        return ""
    # Prefer exact key match.
    for key, block in heading_index:
        if key == q_norm:
            return block
    # Allow near-exact match for multi-token headings.
    q_tokens = content_tokens(q_norm)
    if len(q_tokens) >= 2:
        best_score = 0.0
        best_block = ""
        for key, block in heading_index:
            if not key:
                continue
            k_tokens = content_tokens(key)
            if not k_tokens:
                continue
            if len(k_tokens) == 1 and len(q_tokens) >= 2:
                continue
            sim = SequenceMatcher(None, q_norm, key).ratio()
            overlap = len(set(q_tokens) & set(k_tokens)) / max(1, len(set(q_tokens)))
            score = (0.6 * sim) + (0.4 * overlap)
            if q_norm in key or key in q_norm:
                score += 0.15
            if score > best_score:
                best_score = score
                best_block = block
        if best_score >= 0.80:
            return best_block
    return ""


def split_multi_query(raw_query: str) -> List[str]:
    q = normalize_mojibake(raw_query).lower()
    # Protect common "X and Y" phrases that are single headings.
    protected_phrases = [
        "rules and regulations",
        "rules and regulation",
        "concessions and scholarships",
        "associations and clubs",
        "students' council",
        "students’ council",
        "co-curricular activities",
        "campus facilities and students' amenities",
        "important phone numbers",
    ]
    for phrase in protected_phrases:
        q = q.replace(phrase, phrase.replace(" ", "_"))
    parts: List[str] = []
    # Split on common conjunctions.
    for seg in re.split(r"\s+and\s+|&|/", q):
        seg = seg.strip(" :;,-()[]")
        seg = seg.replace("_", " ")
        seg_clean = re.sub(r"[^a-z0-9]+", "", seg)
        if seg_clean in {"sc", "st", "scc"}:
            continue
        if len(seg) >= 4:
            parts.append(seg)
    return parts


def extract_block_by_query_line(source_text: str, query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm or len(q_norm) < 4:
        return ""

    # Special-case: keep the full Adi Dravidar scholarship subsection.
    if "adi dravidar" in q_norm or "sc st scc" in q_norm or "sc/st/scc" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "b) ADI DRAVIDAR WELFARE SCHOLARSHIP (SC/ST/SCC)",
                "ADI DRAVIDAR WELFARE SCHOLARSHIP (SC/ST/SCC)",
                "b) ADI DRAVIDAR WELFARE SCHOLARSHIP",
                "ADI DRAVIDAR WELFARE SCHOLARSHIP",
            ],
            end_markers=[
                "c) DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "c) DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "d) MOOVALUR RAMAMIRTHAM AMMAIYAR",
            ],
        )
        if block:
            return block

    if "differentlyabled welfare scholarship" in q_norm or "differently abled welfare scholarship" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "c) DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "c) DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLYABLED WELFARE SCHOLARSHIP",
            ],
            end_markers=[
                "d) MOOVALUR RAMAMIRTHAM AMMAIYAR",
                "MOOVALUR RAMAMIRTHAM AMMAIYAR",
                "e) POST MATRIC SCHOLARSHIP",
            ],
        )
        if block:
            return block

    if "merit cum means" in q_norm and "minority" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "b) MERIT CUM MEANS BASED SCHOLARSHIP FOR STUDENTS",
                "b) MERIT CUM MEANS BASED SCHOLARSHIP FOR STUDENTS BELONGING TO MINORITY COMMUNITIES",
                "MERIT CUM MEANS BASED SCHOLARSHIP FOR STUDENTS BELONGING TO MINORITY COMMUNITIES",
            ],
            end_markers=[
                "c) CENTRAL SECTOR SCHEME OF SCHOLARSHIP",
                "CENTRAL SECTOR SCHEME OF SCHOLARSHIP",
            ],
        )
        if block:
            return block

    if "post matric scholarship" in q_norm or "post-matric scholarship" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "a) SCHEME OF 'POST-MATRIC SCHOLARSHIP' FOR STUDENTS",
                "SCHEME OF 'POST-MATRIC SCHOLARSHIP' FOR STUDENTS",
            ],
            end_markers=[
                "b) MERIT CUM MEANS BASED SCHOLARSHIP",
                "MERIT CUM MEANS BASED SCHOLARSHIP",
            ],
        )
        if block:
            return block

    if "diocesan scholarship" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "DIOCESAN SCHOLARSHIP",
            ],
            end_markers=[
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN SCHOLARSHIP",
                "OTHER SCHOLARSHIP",
                "SCHOLARSHIP HELP DESK",
            ],
        )
        if block:
            return block

    if "church workers" in q_norm or "non-teaching staff children" in q_norm or "non teaching staff children" in q_norm:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN SCHOLARSHIP",
            ],
            end_markers=[
                "OTHER SCHOLARSHIP",
                "SCHOLARSHIP HELP DESK",
            ],
        )
        if block:
            return block

    other_scholarship = extract_other_scholarship_subsection(source_text, query)
    if other_scholarship:
        return other_scholarship

    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    # Prefer direct Article heading match (e.g., "ARTICLE-III: Office Bearers...").
    if "article" in q_norm:
        # Exact article heading match with line-based extraction.
        for i, line in enumerate(lines):
            if re.search(r"^article[-\s]*iii\s*:\s*office bearers of the executive committee", line, flags=re.I):
                end_i = len(lines)
                for j in range(i + 1, len(lines)):
                    if re.match(r"^article[-\s]*[ivx]+", lines[j], flags=re.I):
                        end_i = j
                        break
                    if normalize_heading_key(lines[j]) == normalize_heading_key("the composition of the executive committee"):
                        end_i = j
                        break
                return "\n".join(lines[i:end_i]).strip()
        for i, line in enumerate(lines):
            l_norm = normalize_heading_key(line)
            if q_norm == l_norm:
                end_i = len(lines)
                for j in range(i + 1, len(lines)):
                    if is_inline_heading_candidate(lines[j]):
                        end_i = j
                        break
                    if normalize_heading_key(lines[j]) == normalize_heading_key("the composition of the executive committee"):
                        end_i = j
                        break
                return "\n".join(lines[i:end_i]).strip()
        # Fallback: find the article line, then include following lines if they contain the key tokens.
        for i, line in enumerate(lines):
            l_norm = normalize_heading_key(line)
            if l_norm.startswith("article-iii") or l_norm.startswith("article iii") or l_norm.startswith("article-3") or l_norm.startswith("article 3"):
                end_i = len(lines)
                for j in range(i + 1, len(lines)):
                    if is_inline_heading_candidate(lines[j]):
                        end_i = j
                        break
                    if normalize_heading_key(lines[j]) == normalize_heading_key("the composition of the executive committee"):
                        end_i = j
                        break
                return "\n".join(lines[i:end_i]).strip()

    q_tokens = set(content_tokens(q_norm))
    best_i = -1
    best_score = 0.0
    for i, line in enumerate(lines):
        l_norm = normalize_heading_key(line)
        if not l_norm:
            continue
        l_tokens = set(content_tokens(l_norm))
        if q_tokens and l_tokens:
            # Require strong token overlap for multi-token queries.
            overlap = len(q_tokens & l_tokens) / max(1, len(q_tokens))
            if len(q_tokens) >= 2 and (overlap < 0.9 or len(l_tokens) < len(q_tokens)):
                pass_check = False
            else:
                pass_check = True
        else:
            overlap = 0.0
            pass_check = True
        if q_norm == l_norm or q_norm in l_norm or l_norm in q_norm:
            if is_inline_heading_candidate(line):
                score = 3.0 if q_norm == l_norm else 2.0
                if q_tokens and l_tokens and len(l_tokens) < len(q_tokens):
                    score -= 0.4
                if overlap:
                    score += overlap
                if score > best_score:
                    best_score = score
                    best_i = i
        else:
            # Fuzzy match for short headings (e.g., "Mode of Election").
            if len(q_norm) >= 8 and is_inline_heading_candidate(line) and pass_check:
                sim = SequenceMatcher(None, q_norm, l_norm).ratio()
                score = sim + overlap
                if l_tokens and q_tokens and len(l_tokens) < len(q_tokens):
                    score -= 0.3
                if score > 0.86 and score > best_score:
                    best_score = score
                    best_i = i

    if best_i < 0:
        return ""

    end_i = len(lines)
    # If the match is a lettered heading (e.g., "a) ..."), capture until the next
    # lettered heading, not just the next inline heading.
    if is_lettered_heading(lines[best_i]):
        for j in range(best_i + 1, len(lines)):
            if re.match(r"^[a-h]\)\s+", lines[j], flags=re.I):
                end_i = j
                break
            if is_main_heading(lines[j]):
                end_i = j
                break
    else:
        for j in range(best_i + 1, len(lines)):
            if is_inline_heading_candidate(lines[j]):
                end_i = j
                break

    block = "\n".join(lines[best_i:end_i]).strip()
    return strip_leading_heading_line(block)


def extract_named_faculty_block(source_text: str, query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if "faculty" not in q_norm:
        return ""
    q_focus = normalize_heading_key(
        re.sub(
            r"\b(details?|detail|about|information|info|show|tell|give|me|of)\b",
            " ",
            q_norm,
        )
    )
    if not q_focus:
        q_focus = q_norm
    q_tokens = set(content_tokens(q_focus))
    if not q_tokens:
        return ""
    q_has_ug = "ug" in q_tokens
    q_has_pg = "pg" in q_tokens

    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    def _faculty_block_from_index(start_i: int) -> str:
        q_has_sf = any(token in q_focus for token in ("sf", "self financed", "self-financed"))
        q_wants_full_details = any(token in q_norm for token in ("detail", "details", "full"))
        end_i = len(lines)
        for j in range(start_i + 1, len(lines)):
            line_norm = normalize_heading_key(lines[j])
            if not q_has_sf and not q_wants_full_details and line_norm == "self financed":
                end_i = j
                break
            if is_upper_heading(lines[j]) and not re.match(r"^\d", lines[j]):
                end_i = j
                break
            if line_norm == normalize_heading_key("non - teaching staff"):
                end_i = j
                break
        return strip_leading_heading_line("\n".join(lines[start_i:end_i]))

    # Prefer exact faculty-heading match first.
    for i, line in enumerate(lines):
        if "FACULTY" not in line.upper():
            continue
        line_norm = normalize_heading_key(line)
        line_tokens = set(content_tokens(line_norm))
        if q_has_ug and "ug" not in line_tokens:
            continue
        if q_has_pg and "pg" not in line_tokens:
            continue
        if line_norm == q_focus:
            return _faculty_block_from_index(i)

    best_i = -1
    best_score = 0.0
    for i, line in enumerate(lines):
        if "FACULTY" not in line.upper():
            continue
        l_norm = normalize_heading_key(line)
        l_tokens = set(content_tokens(l_norm))
        if not l_tokens:
            continue
        if "faculty" not in l_tokens:
            continue
        if q_has_ug and "ug" not in l_tokens:
            continue
        if q_has_pg and "pg" not in l_tokens:
            continue
        overlap = len(q_tokens & l_tokens) / max(1, len(q_tokens))
        sim = SequenceMatcher(None, q_focus, l_norm).ratio()
        score = (0.65 * overlap) + (0.35 * sim)
        if q_tokens and l_tokens and q_tokens.issubset(l_tokens):
            score += 0.2
        if "sf" in q_focus and "sf" in l_norm:
            score += 0.15
        if score > best_score and score >= 0.55:
            best_score = score
            best_i = i

    if best_i < 0:
        return ""
    return _faculty_block_from_index(best_i)


def extract_endowment_scholarships_block_from_text(source_text: str) -> str:
    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    start = -1
    for i, line in enumerate(lines):
        if normalize_heading_key(line) == normalize_heading_key("ENDOWMENT SCHOLARSHIPS"):
            start = i
            break
    if start < 0:
        return ""

    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if is_upper_heading(ln) and not ln.startswith("DEPARTMENT OF"):
            end = j
            break
    block = "\n".join(lines[start:end]).strip()
    return strip_leading_heading_line(block)


def extract_department_block_from_text(source_text: str, dept_name: str) -> str:
    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    target = normalize_heading_key(dept_name)
    start = -1
    for i, line in enumerate(lines):
        if normalize_heading_key(line) == target:
            start = i
            break
    if start < 0:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if is_upper_heading(ln) and normalize_heading_key(ln).startswith("department of"):
            end = j
            break
    block = "\n".join(lines[start:end]).strip()
    return strip_leading_heading_line(block)


def expand_main_heading_block(source_text: str, heading_line: str) -> str:
    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    target = normalize_heading_key(heading_line)
    if not target:
        return ""
    start = -1
    for i, line in enumerate(lines):
        if normalize_heading_key(line) == target:
            start = i
            break
    if start < 0:
        return ""
    end = len(lines)
    current_num = None
    m = re.match(r"^(\d+)\.\s+", lines[start])
    if m:
        current_num = int(m.group(1))
    if current_num is not None:
        for j in range(start + 1, len(lines)):
            m2 = re.match(r"^(\d+)\.\s+(.+)$", lines[j])
            if m2:
                title = m2.group(2).strip()
                # Stop at next numbered ALL-CAPS heading (e.g., "20. ROOMS INDEX"),
                # not at numbered list items with mixed case.
                if title and (title == title.upper() or is_upper_heading(title)):
                    end = j
                    break
    return "\n".join(lines[start:end]).strip()


def map_query_to_heading_block(query: str, heading_index: List[Tuple[str, str]], source_text: str) -> str:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q or len(q) < 4:
        return ""

    def block_for_key(target_key: str) -> str:
        t = normalize_heading_key(target_key)
        for i, (key, block) in enumerate(heading_index):
            if key == t:
                # If the block is only a heading, and it's a main heading,
                # merge following subheading blocks until the next main heading.
                lines = [x for x in block.splitlines() if x.strip()]
                if lines and is_heading_only_block(block) and is_main_heading(lines[0]):
                    merged = [block.strip()]
                    for j in range(i + 1, len(heading_index)):
                        next_block = heading_index[j][1]
                        next_lines = [x for x in next_block.splitlines() if x.strip()]
                        if next_lines and is_main_heading(next_lines[0]):
                            break
                        merged.append(next_block.strip())
                    merged_block = "\n\n".join([m for m in merged if m])
                    # Trim known bleed-over for specific headings.
                    merged_block = trim_section_by_heading(lines[0], merged_block)
                    return merged_block
                return block
        return ""

    # Explicit fixes for common variants.
    if "issue of certificates" in q or "issue of certificate" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["b) Issue of Certificates", "Issue of Certificates"],
            end_markers=["c) Attendance and Leave Policy", "Attendance and Leave Policy"],
        )
        if block:
            return block
    if ("rules" in q or "regulations" in q) and "general discipline" not in q:
        # Return all sub-sections under Rules & Regulations.
        start_key = normalize_heading_key("a) General Discipline")
        if start_key:
            merged: List[str] = []
            capture = False
            for key, block in heading_index:
                if key == start_key:
                    capture = True
                if capture:
                    lines = [x for x in block.splitlines() if x.strip()]
                    if lines and is_main_heading(lines[0]):
                        # Stop at the next main heading (do not include it).
                        break
                    merged.append(block.strip())
            merged_block = "\n\n".join([m for m in merged if m])
            # Hard stop if next major section markers appear inside.
            cut_markers = [
                "9. STUDENTS' COUNCIL",
                "9. STUDENTSâ€™ COUNCIL",
                "10. OUR UNIVERSITY",
                "10. Our University",
            ]
            cut_at = len(merged_block)
            for marker in cut_markers:
                pos = merged_block.find(marker)
                if pos >= 0:
                    cut_at = min(cut_at, pos)
            merged_block = merged_block[:cut_at].strip()
            return merged_block
    if "issue of certificates" in q:
        return block_for_key("b) Issue of Certificates")
    if "attendance and leave policy" in q:
        return block_for_key("c) Attendance and Leave Policy")
    if "endowment" in q:
        block = extract_endowment_scholarships_block_from_text(source_text)
        if block:
            return block
        return block_for_key("endowment scholarships")
    if "associations and clubs" in q:
        # Return full Associations & Clubs section until next main heading.
        lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
        lines = [x for x in lines if x and not x.startswith("--- Page")]
        start = -1
        for i, line in enumerate(lines):
            if normalize_heading_key(line) == normalize_heading_key("14. ASSOCIATIONS AND CLUBS"):
                start = i
                break
        if start >= 0:
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if re.match(r"^\d+\.\s+", lines[j]):
                    end = j
                    break
            return "\n".join(lines[start:end]).strip()
    if "utility services" in q:
        block = expand_main_heading_block(source_text, "16. UTILITY SERVICES")
        if block:
            return block
        block = expand_main_heading_block(source_text, "UTILITY SERVICES")
        if block:
            return block
    if "colleges affiliated" in q or ("m s university" in q and "college" in q):
        block = expand_main_heading_block(source_text, "19. COLLEGES AFFILIATED TO M.S UNIVERSITY")
        if block:
            return block
        block = expand_main_heading_block(source_text, "19. COLLEGES AFFILIATED TO M.S. UNIVERSITY")
        if block:
            return block
        block = expand_main_heading_block(source_text, "COLLEGES AFFILIATED TO M.S UNIVERSITY")
        if block:
            return block
    if "rooms index" in q:
        block = expand_main_heading_block(source_text, "20. ROOMS INDEX")
        if block:
            return block
        block = expand_main_heading_block(source_text, "ROOMS INDEX")
        if block:
            return block
    if "teaching staff" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["TEACHING STAFF"],
            end_markers=[
                "NON - TEACHING STAFF",
                "NON-TEACHING STAFF",
                "NON TEACHING STAFF",
                "4. COURSES OFFERED",
                "4. Courses Offered",
            ],
        )
        if block:
            return block
    if "composition of the executive committee" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["The Composition of the Executive Committee"],
            end_markers=["Article-IV : Election", "Article-IV: Election", "ARTICLE-IV : Election", "ARTICLE-IV: Election"],
        )
        if block:
            return block
    if "mode of election" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["Mode of Election"],
            end_markers=["Eligibility to Contest", "Eligibility to Contest :"],
        )
        if block:
            return block
    if "eligibility to contest" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["Eligibility to Contest", "Eligibility to Contest :"],
            end_markers=[],
        )
        if block:
            return block
    if "semester" in q:
        block = expand_main_heading_block(source_text, "11. CURRICULUM")
        if block:
            return block
        block = expand_main_heading_block(source_text, "CURRICULUM")
        if block:
            return block
    if "academic calendar" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "17. NMCC ACADEMIC CALENDAR 2024 - 2025",
                "NMCC ACADEMIC CALENDAR 2024 - 2025",
            ],
            end_markers=[
                "18. IMPORTANT PHONE NUMBERS",
                "18. Important Phone Numbers",
            ],
        )
        if block:
            return block
    if "college song" in q:
        block = extract_college_song_block(source_text)
        if block:
            return block
    if "staff council members" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["STAFF COUNCIL MEMBERS 2024-25"],
            end_markers=["3. MEMBERS OF THE STAFF"],
        )
        if block:
            return block
    if "teaching staff" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["TEACHING STAFF"],
            end_markers=["NON - TEACHING STAFF", "NON-TEACHING STAFF"],
        )
        if block:
            return block
    if "courses offered" in q or "courses offerd" in q or "courses" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["4. COURSES OFFERED"],
            end_markers=[
                "Tamil 42. Dr.",
                "5. Ph.D. GUIDES",
                "5. Ph.D. Guides",
                "Ph.D. GUIDES",
                "5. PH.D. GUIDES",
            ],
        )
        if block:
            filtered = extract_courses_for_query(block, q)
            return filtered or block
    if "ug aided" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["U.G. Aided"],
            end_markers=["U.G. Self - Financing Scheme", "U.G. Self- Financing Scheme"],
        )
        if block:
            return block
    if "ug self" in q or "ug self financing" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["U.G. Self - Financing Scheme", "U.G. Self- Financing Scheme"],
            end_markers=["P.G. Aided"],
        )
        if block:
            return block
    if "pg aided" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["P.G. Aided"],
            end_markers=["P.G. Self - Financing Scheme", "P.G. Self- Financing Scheme"],
        )
        if block:
            return block
    if "pg self" in q or "pg self financing" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["P.G. Self - Financing Scheme", "P.G. Self- Financing Scheme"],
            end_markers=["Ph.D."],
        )
        if block:
            return block
    if q.strip() == "ph.d" or "ph.d" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["Ph.D."],
            end_markers=["CAREER ORIENTED PROGRAMMES - UNIVERSITY APPROVED"],
        )
        if block:
            return block
    if "career oriented" in q or "certificate" in q or "diploma" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["CAREER ORIENTED PROGRAMMES - UNIVERSITY APPROVED"],
            end_markers=["Training Programmes conducted by College"],
        )
        if block:
            return block
    if "training programmes" in q or "training programs" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["Training Programmes conducted by College"],
            end_markers=["5. PH.D. GUIDES", "5. Ph.D. GUIDES", "Ph.D. GUIDES"],
        )
        if block:
            return block
    if "ph.d. guides" in q or "ph d guides" in q or "phd guides" in q:
        block = extract_phd_guides_block(source_text)
        if block:
            return block
    if "state government scholarships" in q or "state government scholarship" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["STATE GOVERNMENT SCHOLARSHIPS"],
            end_markers=["NATIONAL SCHOLARSHIPS", "National Scholarships"],
        )
        if block:
            return block
    if "adi dravidar" in q or "sc st scc" in q or "sc/st/scc" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "b) ADI DRAVIDAR WELFARE SCHOLARSHIP (SC/ST/SCC)",
                "ADI DRAVIDAR WELFARE SCHOLARSHIP (SC/ST/SCC)",
                "b) ADI DRAVIDAR WELFARE SCHOLARSHIP",
                "ADI DRAVIDAR WELFARE SCHOLARSHIP",
            ],
            end_markers=[
                "c) DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "c) DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "d) MOOVALUR RAMAMIRTHAM AMMAIYAR",
            ],
        )
        if block:
            return block
    if "differentlyabled welfare scholarship" in q or "differently abled welfare scholarship" in q:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=["DIFFERENTLY ABLED WELFARE SCHOLARSHIP"],
            end_markers=["d) MOOVALUR RAMAMIRTHAM AMMAIYAR", "MOOVALUR RAMAMIRTHAM AMMAIYAR"],
        )
        if block:
            return block
    if q.startswith("department of"):
        block = extract_department_block_from_text(source_text, q)
        if block:
            return block
    if "general discipline" in q:
        return block_for_key("a) General Discipline")

    q_tokens = set(content_tokens(q))
    for key, _ in heading_index:
        if not key:
            continue
        if key in q:
            return block_for_key(key)
        k_tokens = set(content_tokens(key))
        if k_tokens and k_tokens.issubset(q_tokens):
            return block_for_key(key)
    return ""


def extract_teaching_staff_response(source_text: str, query: str) -> str:
    named_faculty = extract_named_faculty_block(source_text, query)
    if named_faculty:
        return clean_output_text(named_faculty)

    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["TEACHING STAFF"],
        end_markers=[
            "NON - TEACHING STAFF",
            "NON-TEACHING STAFF",
            "NON TEACHING STAFF",
            "4. COURSES OFFERED",
            "4. Courses Offered",
        ],
    )
    return clean_output_text(block) if block else ""


def extract_university_query_block(source_text: str) -> str:
    return extract_block_between_heading_lines(
        source_text=source_text,
        start_patterns=[
            r"^\s*10\.\s+OUR UNIVERSTIY\s*$",
            r"^\s*10\.\s+OUR UNIVERSITY\s*$",
            r"^\s*OUR UNIVERSITY\s*$",
        ],
        end_patterns=[
            r"^\s*11\.\s+CURRICULUM\s*$",
            r"^\s*11\.\s+CO[-\s]CURRICULAR ACTIVITIES\s*$",
            r"^\s*12\.\s+CO[-\s]CURRICULAR ACTIVITIES\s*$",
        ],
    )


def answer_query(
    query: str,
    source_text: str,
    heading_sections: List[Tuple[str, str]],
    subheading_sections: List[Tuple[str, str]],
    heading_index: List[Tuple[str, str]],
    intents: List[Dict[str, object]],
) -> str:
    def _maybe_filter_courses(answer: str) -> str:
        if not answer:
            return answer
        if is_courses_query(query):
            filtered = extract_courses_for_query(answer, query)
            return filtered or answer
        return answer

    def _clean(block: str) -> str:
        return clean_output_text(block) if block else ""

    def _extract_with_fallback(extractor) -> str:
        block = extractor(source_text)
        if not block:
            block = extractor(load_extracted_text_fallback())
        return _clean(block)

    intent_response, intent_score = find_best_intent_match(query, intents)
    other_scholarship_block = extract_other_scholarship_subsection(source_text, query)
    main_heading_block = extract_main_heading_block(source_text, query)
    strong_intent_match = bool(intent_response and intent_score >= 1.8)
    q_norm = normalize_heading_key(normalize_mojibake(query))
    asks_teaching_staff = any(w in q_norm for w in ("faculty", "faculties", "teacher", "teachers")) or "teaching staff" in q_norm

    if is_greeting_query(query):
        return GREETING_REPLY
    elif is_bye_query(query):
        return BYE_REPLY
    elif q_norm == "college":
        block = extract_history_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
        block = extract_about_nmcc_block(source_text)
        if block:
            return clean_output_text(block)
        return intent_response if intent_response else UNKNOWN_REPLY
    elif normalize_heading_key(normalize_mojibake(query)) == "scholarship help desk":
        return (
            "Principal 9443370257 principalnmcc2014@gmail.com\n"
            "Nodal Officer\n"
            "Dr. P.C. Jose Paul 9443000251 pcjosepaul@gmail.com\n"
            "Scholarship Section\n"
            "Mr. G. Lazer 9442844909 lazernmcc@gmail.com"
        )
    elif is_all_scholarships_query(query):
        block = extract_scholarship_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
    elif normalize_heading_key(normalize_mojibake(query)) == "tamilpudhalvan scheme":
        return (
            "Eligibility Criteria\n"
            "All male students who enrolled in Tamil Medium from class VI\n"
            "to XII from government and government aided schools who are\n"
            "pursuing higher education."
        )
    elif strong_intent_match:
        return intent_response
    elif is_unknown_or_gibberish_query(query):
        return UNKNOWN_REPLY
    elif (
        "church workers" in normalize_heading_key(normalize_mojibake(query))
        or "non teaching staff children" in normalize_heading_key(normalize_mojibake(query))
        or "non-teaching staff children" in normalize_heading_key(normalize_mojibake(query))
        or (
            {"church", "workers", "non", "teaching", "staff"}.issubset(set(content_tokens(query)))
            and (
                "children" in set(content_tokens(query))
                or "childrensscholarship" in set(content_tokens(query))
                or "childrens" in set(content_tokens(query))
            )
        )
    ):
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON TEACHING STAFF CHILDREN'S SCHOLARSHIP",
                "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN SCHOLARSHIP",
            ],
            end_markers=[
                "OTHER SCHOLARSHIP",
                "SCHOLARSHIP HELP DESK",
            ],
        )
        if block:
            return clean_output_text(block)
    elif other_scholarship_block:
        return clean_output_text(other_scholarship_block)
    elif is_non_teaching_staff_query(query):
        block = extract_non_teaching_staff_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
    elif is_courses_main_heading_query(query):
        block = extract_courses_full_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
    elif is_concessions_and_scholarships_query(query):
        block = extract_concessions_and_scholarships_intro_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_history_query(query):
        block = extract_history_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
    elif is_about_nmcc_query(query):
        block = extract_history_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
        block = extract_about_nmcc_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_staff_council_members_query(query):
        block = extract_staff_council_members_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_staff_council_query(query):
        block = extract_staff_council_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_campus_facilities_query(query):
        answer = _extract_with_fallback(extract_campus_facilities_block)
        if answer:
            return answer
    elif is_rules_regulations_main_query(query):
        answer = _extract_with_fallback(extract_rules_regulations_block)
        if answer:
            return answer
    elif is_utility_services_query(query):
        answer = _extract_with_fallback(extract_utility_services_block)
        if answer:
            return answer
    elif is_important_phone_numbers_query(query):
        answer = _extract_with_fallback(extract_important_phone_numbers_block)
        if answer:
            return answer
    elif is_state_scholarship_subquery(query):
        subkey = is_state_scholarship_subquery(query)
        block = extract_state_scholarship_subsection(source_text, subkey)
        if block:
            return clean_output_text(block)
    elif is_ugc_scholarship_subquery(query):
        subkey = is_ugc_scholarship_subquery(query)
        block = extract_ugc_scholarship_subsection(source_text, subkey)
        if block:
            return clean_output_text(block)
    elif is_ugc_scholarships_query(query):
        block = extract_ugc_scholarships_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_colleges_affiliated_query(query):
        answer = _extract_with_fallback(extract_colleges_affiliated_block)
        if answer:
            return answer
    else:
        direct_subheading = find_subheading_block_in_text(source_text, query)
        if direct_subheading:
            return clean_output_text(direct_subheading)
        subheading_block = match_subheading_section(query, subheading_sections)
        if subheading_block:
            return clean_output_text(strip_leading_heading_line(subheading_block))
    if is_scholarship_help_desk_query(query):
        block = extract_scholarship_help_desk_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_university_scholarship_query(query):
        block = extract_university_scholarship_block(source_text)
        if block:
            return clean_output_text(block)
    elif asks_teaching_staff:
        answer = extract_teaching_staff_response(source_text, query)
        if answer:
            return answer
    elif main_heading_block:
        return clean_output_text(main_heading_block)
    elif is_national_scholarship_subquery(query):
        subkey = is_national_scholarship_subquery(query)
        block = extract_national_scholarship_subsection(source_text, subkey)
        if block:
            return clean_output_text(block)
    elif is_national_scholarship_query(query):
        block = extract_national_scholarships_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_state_government_scholarship_query(query):
        block = extract_state_government_scholarships_block(source_text)
        if block:
            return clean_output_text(block)
    elif is_university_query(query):
        if "rules" in q_norm or "regulations" in q_norm:
            pass
        else:
            block = extract_university_query_block(source_text)
            if block:
                return clean_output_text(block)
            return (
                "Sorry, the Our University section isn't present in the extracted text. "
                "Please re-run py1.py to re-extract the PDF, then try again."
            )
    elif "rules" in q_norm or "regulations" in q_norm:
        block = extract_block_between_heading_lines(
            source_text=source_text,
            start_patterns=[
                r"^\s*8\.\s+RULES AND REGULATIONS\s*$",
                r"^\s*8\.\s+RULES\s*&\s*REGULATIONS\s*$",
                r"^\s*RULES AND REGULATIONS\s*$",
                r"^\s*RULES\s*&\s*REGULATIONS\s*$",
                r"^\s*a\)\s+General Discipline\s*$",
            ],
            end_patterns=[
                r"^\s*9\.\s+STUDENTS['’]\s+COUNCIL\s*$",
                r"^\s*10\.\s+OUR UNIVERSTIY\s*$",
                r"^\s*10\.\s+OUR UNIVERSITY\s*$",
            ],
        )
        if block:
            return clean_output_text(block)
        return (
            "Sorry, the Rules and Regulations section isn't present in the extracted text. "
            "Please re-run py1.py to re-extract the PDF, then try again."
        )
    elif is_scholarship_query(query) and not is_specific_scholarship_query(query):
        block = extract_scholarship_block(source_text, heading_sections)
        if block:
            return clean_output_text(block)
    else:
        q_norm = normalize_heading_key(normalize_mojibake(query))
        if "semester" in q_norm:
            mapped = map_query_to_heading_block("curriculum", heading_index, source_text)
            if mapped:
                cleaned = clean_output_text(mapped)
                if not is_heading_only_block(cleaned):
                    return _maybe_filter_courses(cleaned)
        if "composition of the executive committee" in q_norm:
            block = extract_block_between_markers(
                source_text=source_text,
                start_markers=["The Composition of the Executive Committee"],
                end_markers=["Article-IV : Election", "Article-IV: Election", "ARTICLE-IV : Election", "ARTICLE-IV: Election"],
            )
            if block:
                return clean_output_text(block)
        q_focus = extract_query_focus(query)
        multi_parts = split_multi_query(query)
        if len(multi_parts) >= 2:
            blocks = []
            missing = []
            for part in multi_parts[:3]:
                part_norm = normalize_heading_key(part)
                blk = ""
                if "rules" in part_norm or "regulations" in part_norm or "regulation" in part_norm:
                    blk = extract_block_between_heading_lines(
                        source_text=source_text,
                        start_patterns=[
                            r"^8\.\s+RULES AND REGULATIONS\s*$",
                            r"^8\.\s+RULES\s*&\s*REGULATIONS\s*$",
                            r"^RULES AND REGULATIONS\s*$",
                            r"^RULES\s*&\s*REGULATIONS\s*$",
                            r"^a\)\s+General Discipline\s*$",
                        ],
                        end_patterns=[
                            r"^9\.\s+STUDENTS['’]\s+COUNCIL\s*$",
                            r"^10\.\s+OUR UNIVERSTIY\s*$",
                            r"^10\.\s+OUR UNIVERSITY\s*$",
                        ],
                    )
                elif "university" in part_norm:
                    blk = extract_block_between_heading_lines(
                        source_text=source_text,
                        start_patterns=[
                            r"^10\.\s+OUR UNIVERSTIY\s*$",
                            r"^10\.\s+OUR UNIVERSITY\s*$",
                            r"^OUR UNIVERSITY\s*$",
                        ],
                        end_patterns=[
                            r"^11\.\s+CURRICULUM\s*$",
                            r"^11\.\s+CO[-\s]CURRICULAR ACTIVITIES\s*$",
                            r"^12\.\s+CO[-\s]CURRICULAR ACTIVITIES\s*$",
                        ],
                    )
                if not blk:
                    blk = extract_block_by_query_line(source_text, part)
                if not blk:
                    blk = exact_heading_match(part, heading_index)
                if blk:
                    blocks.append(clean_output_text(blk))
                else:
                    missing.append(part)
            if blocks:
                output = "\n\n".join(blocks)
                if missing:
                    output = f"{output}\n\nMissing sections: {', '.join(missing[:3])}."
                return _maybe_filter_courses(output)
            # If we attempted a multi-part query, do not fall back to a single-word
            # heading match that can capture generic terms like "eligibility".
            multi_fallback_guard = True
        else:
            multi_fallback_guard = False
        exact = exact_heading_match(q_focus or query, heading_index)
        if exact:
            cleaned = clean_output_text(exact)
            if not is_heading_only_block(cleaned):
                return _maybe_filter_courses(cleaned)
        inline = extract_block_by_query_line(source_text, q_focus or query)
        if inline:
            cleaned = clean_output_text(inline)
            if not is_heading_only_block(cleaned):
                return _maybe_filter_courses(cleaned)
        if multi_fallback_guard:
            return UNKNOWN_REPLY
        if q_focus and q_focus in SYNONYM_MAP:
            for alt in expand_query_synonyms(q_focus):
                mapped = map_query_to_heading_block(alt, heading_index, source_text)
                if mapped:
                    cleaned = clean_output_text(mapped)
                    if not is_heading_only_block(cleaned):
                        return _maybe_filter_courses(cleaned)
        if q_focus and q_focus != q_norm:
            mapped = map_query_to_heading_block(q_focus, heading_index, source_text)
            if mapped:
                cleaned = clean_output_text(mapped)
                if not is_heading_only_block(cleaned):
                    return _maybe_filter_courses(cleaned)
            for alt in expand_query_synonyms(q_focus):
                mapped = map_query_to_heading_block(alt, heading_index, source_text)
                if mapped:
                    cleaned = clean_output_text(mapped)
                    if not is_heading_only_block(cleaned):
                        return _maybe_filter_courses(cleaned)
        if q_norm:
            for alt in expand_query_synonyms(q_norm):
                mapped = map_query_to_heading_block(alt, heading_index, source_text)
                if mapped:
                    cleaned = clean_output_text(mapped)
                    if not is_heading_only_block(cleaned):
                        return _maybe_filter_courses(cleaned)

        mapped_block = map_query_to_heading_block(query, heading_index, source_text)
        if mapped_block:
            cleaned = clean_output_text(mapped_block)
            if not is_heading_only_block(cleaned):
                return _maybe_filter_courses(cleaned)

        heading_answer = auto_heading_match(query, heading_index)
        if heading_answer:
            cleaned = clean_output_text(heading_answer)
            if not is_heading_only_block(cleaned):
                return _maybe_filter_courses(cleaned)

        direct_subheading = find_subheading_block_in_text(source_text, query)
        if direct_subheading:
            return _maybe_filter_courses(clean_output_text(direct_subheading))

        direct_heading = extract_heading_block_by_query(source_text, query)
        if direct_heading:
            cleaned = clean_output_text(direct_heading)
            if is_heading_only_block(cleaned):
                lines = [x for x in cleaned.splitlines() if x.strip()]
                if lines and is_main_heading(lines[0]):
                    expanded = expand_main_heading_block(source_text, lines[0])
                    if expanded:
                        return _maybe_filter_courses(clean_output_text(expanded))
            return _maybe_filter_courses(cleaned)

    if intent_response:
        return intent_response

    return UNKNOWN_REPLY


def build_heading_sections(source_text: str) -> List[Tuple[str, str]]:
    raw_lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in raw_lines if x]

    toc_titles = extract_toc_headings(source_text)
    if toc_titles:
        toc_keys = {normalize_heading_key(t): t for t in toc_titles}
        starts: List[Tuple[int, int, str]] = []

        for i, line in enumerate(lines):
            if line.startswith("--- Page"):
                continue
            if is_toc_line(line):
                continue
            parsed = parse_main_heading_line(line)
            if not parsed:
                continue
            num, title = parsed
            key = normalize_heading_key(title)
            if key in toc_keys and 1 <= num <= 30:
                starts.append((i, num, toc_keys[key]))

        if starts:
            starts = sorted(starts, key=lambda x: x[0])
            sections: List[Tuple[str, str]] = []
            for idx, (start_i, num, canon_title) in enumerate(starts):
                end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
                block_lines = [x for x in lines[start_i:end_i] if not x.startswith("--- Page")]
                block = "\n".join(block_lines).strip()
                heading = f"{num}. {canon_title}"
                block = trim_section_by_heading(heading, block)
                if block:
                    sections.append((heading, block))
            if sections:
                return sections

    heads: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if line.startswith("--- Page"):
            continue
        if is_main_heading(line):
            heads.append((i, line))

    sections: List[Tuple[str, str]] = []
    if not heads:
        return sections

    for idx, (start_i, heading) in enumerate(heads):
        end_i = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        block_lines = [x for x in lines[start_i:end_i] if not x.startswith("--- Page")]
        block = "\n".join(block_lines).strip()
        block = trim_section_by_heading(heading, block)
        if block:
            sections.append((heading, block))
    return sections


def extract_heading_block_by_query(source_text: str, query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm:
        return ""

    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    q_tokens = set(content_tokens(q_norm))
    best_i = -1
    best_score = 0.0
    for i, line in enumerate(lines):
        if is_toc_line(line):
            continue
        if not is_any_heading_line(line):
            continue
        l_norm = normalize_heading_key(line)
        if not l_norm:
            continue
        l_tokens = set(content_tokens(l_norm))

        score = 0.0
        if l_norm == q_norm:
            score = 3.0
        elif q_norm in l_norm:
            score = 2.0
        elif l_norm in q_norm:
            score = 1.8

        if q_tokens and l_tokens:
            overlap = len(q_tokens & l_tokens) / max(1, len(q_tokens))
            score += overlap
            if q_tokens.issubset(l_tokens):
                score += 0.4
            if len(l_tokens) < len(q_tokens):
                score -= 0.2

        if score > best_score:
            best_score = score
            best_i = i

    if best_i < 0:
        return ""

    end_i = len(lines)
    for i in range(best_i + 1, len(lines)):
        if is_any_heading_line(lines[i]) and not is_toc_line(lines[i]):
            end_i = i
            break

    block = "\n".join(lines[best_i:end_i]).strip()
    # If the matched heading has no body, try returning the immediate child heading block
    # (e.g., "UNIVERSITY SCHOLARSHIP" -> "UNIVERSITY MERIT SCHOLARSHIP").
    block_lines = [x for x in block.splitlines() if x and not x.startswith("--- Page")]
    if len(block_lines) <= 1:
        if end_i < len(lines):
            parent_norm = normalize_heading_key(block_lines[0]) if block_lines else ""
            child_line = lines[end_i]
            child_norm = normalize_heading_key(child_line)
            parent_tokens = set(content_tokens(parent_norm))
            child_tokens = set(content_tokens(child_norm))
            if parent_tokens and parent_tokens.issubset(child_tokens):
                next_end = len(lines)
                for j in range(end_i + 1, len(lines)):
                    if is_any_heading_line(lines[j]) and not is_toc_line(lines[j]):
                        next_end = j
                        break
                block = "\n".join(lines[end_i:next_end]).strip()
    return block


def extract_courses_from_section(section_text: str) -> str:
    raw = normalize_mojibake(section_text)
    # Guard against OCR/section bleed from later headings.
    bleed_markers = [
        "5. PH.D. GUIDES",
        "5. Ph.D. Guides",
        "PH.D. GUIDES",
        "6. FEE, CONCESSIONS & SCHOLARSHIPS",
        "6. Fee, Concessions & Scholarships",
    ]
    cut_at = len(raw)
    for marker in bleed_markers:
        pos = raw.find(marker)
        if pos > 0:
            cut_at = min(cut_at, pos)
    raw = raw[:cut_at]

    lines = [normalize_text(normalize_mojibake(x)) for x in raw.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if not lines:
        return ""

    out: List[str] = []
    seen = set()
    in_course_zone = False
    for ln in lines:
        l = ln.lower()
        if re.search(r"\bdr\.?\b", l):
            # Prevent Ph.D. Guides/staff names from entering courses output.
            continue

        if (
            "courses offered" in l
            or "career oriented programmes" in l
            or "certificate, diploma and advanced diploma courses" in l
            or "training programmes conducted by college" in l
            or "aided stream" in l
            or "self finance stream" in l
            or "self-finance stream" in l
            or "under graduate" in l
            or "post graduate" in l
        ):
            in_course_zone = True
            keep = True
        elif not in_course_zone:
            keep = False
        elif re.match(r"^\d+\.\s+", l):
            keep = bool(
                re.search(
                    r"\b(b\.?a|b\.?sc|b\.?com|b\.?b\.?a|b\.?c\.?a|m\.?a|m\.?sc|m\.?com|mca|mba|ph\.?d)\b",
                    l,
                )
                or any(
                    k in l
                    for k in [
                        "air-ticketing",
                        "business communication",
                        "computational biology",
                        "computer aided accounting",
                        "entrepreneurship",
                        "export and import management",
                        "graphics for visual communication",
                        "handicrafts",
                        "herbal science",
                        "journalism",
                        "spoken english",
                        "visual communication",
                        "spoken hindi",
                        "driving",
                        "tamil",
                        "english",
                        "history",
                        "economics",
                        "mathematics",
                        "physics",
                        "computer science",
                        "chemistry",
                        "botany",
                        "zoology",
                        "commerce",
                        "management studies",
                    ]
                )
            )
        elif re.match(
            r"^(u\.?g\.?|p\.?g\.?|ph\.?d\.?)\s*[-:]", l
        ):
            keep = True
        else:
            keep = False

        if keep:
            key = l
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)

    return "\n".join(out).strip()


def extract_courses_full_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    raw = normalize_mojibake(source_text)
    courses_section = find_section_by_heading_keyword(sections, "courses offered")

    start_markers = [
        "U.G. Aided",
        "UG Aided",
        "U.G Aided",
    ]
    start = -1
    for marker in start_markers:
        pos = raw.find(marker)
        if pos >= 0:
            start = pos
            break

    if start < 0 and courses_section:
        raw = normalize_mojibake(courses_section)
        for marker in start_markers:
            pos = raw.find(marker)
            if pos >= 0:
                start = pos
                break
    if start < 0:
        return ""

    end_markers = [
        "5. PH.D. GUIDES",
        "5. Ph.D. GUIDES",
        "5. Ph.D. Guides",
        "6. FEE, CONCESSIONS & SCHOLARSHIPS",
        "6. Fee, Concessions & Scholarships",
        "Tamil 42. Dr.",
        "1. Dr. A. Sajan 43. Dr.",
    ]
    end = len(raw)
    for marker in end_markers:
        pos = raw.find(marker, start + 1)
        if pos >= 0:
            end = min(end, pos)

    body = raw[start:end]
    lines = [normalize_text(x) for x in body.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    out: List[str] = []
    for ln in lines:
        if re.match(r"^\d{1,3}\.\s*Dr\.?", ln, flags=re.I):
            break
        if re.match(
            r"^(Tamil|English|History|Economics|Mathematics|Physics|Computer Science|Chemistry|Botany|Zoology|Commerce|Management Studies)\s+\d{1,3}\.\s*Dr\.?",
            ln,
            flags=re.I,
        ):
            break
        out.append(ln)

    joined = "\n".join(out).lower()
    if (
        "training programmes conducted by college" in joined
        and "spoken hindi" in joined
        and "driving" not in joined
    ):
        out.append("2. Driving")

    return "\n".join(out).strip()


def extract_courses_for_query(courses_text: str, query: str) -> str:
    text = normalize_mojibake(courses_text).strip()
    if not text:
        return ""

    q = normalize_heading_key(normalize_mojibake(query))

    lines = [normalize_text(x) for x in text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    sections_map: Dict[str, List[str]] = {
        "ug_aided": [],
        "ug_self": [],
        "pg_aided": [],
        "pg_self": [],
        "phd": [],
        "career": [],
        "training": [],
    }

    def _section_key(ln: str) -> str:
        l = normalize_heading_key(ln)
        if "u g" in l and "aided" in l:
            return "ug_aided"
        if "u g" in l and ("self" in l or "financing" in l or "finance" in l):
            return "ug_self"
        if "p g" in l and "aided" in l:
            return "pg_aided"
        if "p g" in l and ("self" in l or "financing" in l or "finance" in l):
            return "pg_self"
        if "ph d" in l or l.strip() == "phd":
            return "phd"
        if "career oriented" in l or "certificate diploma" in l:
            return "career"
        if "training programmes conducted by college" in l:
            return "training"
        return ""

    current = ""
    for ln in lines:
        key = _section_key(ln)
        if key:
            current = key
            sections_map[current] = [ln]
            continue
        if current:
            sections_map[current].append(ln)

    has_any = any(sections_map[k] for k in sections_map)
    if not has_any:
        return text

    wants_ug = bool(re.search(r"\bug\b|u\.?\s*g\.?", q))
    wants_pg = bool(re.search(r"\bpg\b|p\.?\s*g\.?", q))
    wants_phd = bool(re.search(r"\bph\.?\s*d\b", q))
    wants_aided = "aided" in q
    wants_self = any(x in q for x in ["self", "self finance", "self-financing", "self financing", "financing", "sf"])
    wants_career = any(x in q for x in ["career oriented", "certificate", "diploma"])
    wants_training = "training" in q or "spoken hindi" in q or "driving" in q

    selected: List[str] = []
    if wants_ug and wants_aided:
        selected = sections_map.get("ug_aided", [])
    elif wants_ug and wants_self:
        selected = sections_map.get("ug_self", [])
    elif wants_ug:
        selected = sections_map.get("ug_aided", []) + [""] + sections_map.get("ug_self", [])
    elif wants_pg and wants_aided:
        selected = sections_map.get("pg_aided", [])
    elif wants_pg and wants_self:
        selected = sections_map.get("pg_self", [])
    elif wants_pg:
        selected = sections_map.get("pg_aided", []) + [""] + sections_map.get("pg_self", [])
    elif wants_phd:
        selected = sections_map.get("phd", [])
    elif wants_career and wants_training:
        selected = sections_map.get("career", []) + [""] + sections_map.get("training", [])
    elif wants_career:
        selected = sections_map.get("career", [])
    elif wants_training:
        selected = sections_map.get("training", [])

    if selected:
        return "\n".join([x for x in selected if x is not None]).strip()
    return text


def find_section_by_heading_keyword(sections: List[Tuple[str, str]], keyword: str) -> str:
    k = normalize_heading_key(keyword)
    for heading, block in sections:
        h = normalize_heading_key(heading)
        h_title = normalize_heading_key(re.sub(r"^\d+\.\s*", "", heading))
        if k in h or k == h_title:
            return block
        if "course" in k and ("course" in h or "course" in h_title):
            return block
    return ""


def is_pledge_query(query: str) -> bool:
    q = normalize_mojibake(query).lower()
    return bool(re.search(r"\bpledge\b", q))


def is_national_anthem_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "national anthem" in q
        or q.strip() == "anthem"
        or "jana gana mana" in q
    )


def is_college_song_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "college song" in q
        or "song" == q.strip()
        or "we love this college" in q
    )


def is_lord_prayer_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "lord prayer" in q
        or "lords prayer" in q
        or "the lord's prayer" in q
        or "our father" in q
    )


def is_courses_query(query: str) -> bool:
    q = normalize_mojibake(query).lower()
    if "faculty" in q:
        return False
    basic = bool(
        re.search(
            r"\bcourse(s)?\b|\bprogramme(s)?\b|\bprogram(s)?\b|career oriented|certificate|training programme",
            q,
        )
    )
    level_based = bool(
        re.search(r"\b(u\.?\s*g\.?|p\.?\s*g\.?|ph\.?\s*d)\b", q)
        and re.search(r"\b(offer|offered|available|study|studies|aided|self|financing|sf)\b", q)
    )
    return basic or level_based


def is_courses_main_heading_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q in {
        "course offered",
        "courses offered",
        "course offers",
        "courses",
    }


def is_courses_overview_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q in {
            "what courses are available",
            "what are the courses available",
            "courses available",
            "what can i study here",
            "programs offered",
            "list of courses",
        }
        or ("course" in q and "available" in q)
        or ("program" in q and "offered" in q)
    )


def is_concessions_and_scholarships_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q in {
        "concessions and scholarships",
        "concession and scholarship",
        "concession and scholarships",
        "scholarships and concessions",
    }


def is_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q:
        return False
    scholarship_terms = [
        "scholarship",
        "scholarships",
        "fee concession",
        "fee concessions",
        "fee waiver",
        "fee waivers",
        "financial aid",
        "financial assistance",
        "fee relief",
        "tuition concession",
        "tuition waiver",
    ]
    return any(term in q for term in scholarship_terms)


def is_all_scholarships_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q in {
        "what are the scholarships available",
        "what scholarships are available",
        "scholarships available",
        "available scholarships",
        "list of scholarships available",
        "show scholarships available",
        "show me the scholarships available",
    }


def is_state_government_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "state government scholarship" in q
        or "state government scholarships" in q
        or "state scholarship" in q
    )


def is_national_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "national scholarship" in q
        or "national scholarships" in q
    )


def is_ugc_scholarships_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "scholarships by university grants commission" in q
        or "university grants commission scholarships" in q
        or q.strip() == "ugc scholarships"
    )


def is_ugc_scholarship_subquery(query: str) -> str:
    q = normalize_heading_key(normalize_mojibake(query))
    if "post graduate scholarship for single girl child" in q:
        return "POST GRADUATE SCHOLARSHIP FOR SINGLE GIRL CHILD"
    if "postgraduate scholarship for university rank holders" in q or "post graduate scholarship for university rank holders" in q:
        return "POST GRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS"
    return ""


def is_national_scholarship_subquery(query: str) -> str:
    q = normalize_heading_key(normalize_mojibake(query))
    q_compact = q.replace(" ", "")
    if (
        "central sector scheme of scholarship for students with disabilities" in q
        or "centralsectorschemeofscholarshipforstudentswithdisabilities" in q_compact
        or "centralsectorschemeofscholarshipforstudentswithdisabilities" in q_compact.replace("studentswith", "studentswith")
    ):
        return "CENTRAL SECTOR SCHEME OF SCHOLARSHIP FOR STUDENTS WITH DISABILITIES"
    if (
        "national fellowship scholarship for higher education of scheduled tribe students" in q
        or "national fellowship scholarship for higher" in q
        or "scheduled tribe students" in q
    ):
        return "NATIONAL FELLOWSHIP & SCHOLARSHIP FOR HIGHER EDUCATION OF SCHEDULED TRIBE STUDENTS"
    return ""


def is_state_scholarship_subquery(query: str) -> str:
    q = normalize_heading_key(normalize_mojibake(query))
    if "bc mbc dnc" in q or "bc/mbc/dnc" in q:
        return "BC/MBC/DNC SCHOLARSHIPS"
    if "adi dravidar" in q or "sc st scc" in q:
        return "ADI DRAVIDAR WELFARE SCHOLARSHIP"
    if "differentlyabled welfare scholarship" in q or "differently abled welfare scholarship" in q:
        return "DIFFERENTLYABLED WELFARE SCHOLARSHIP"
    if "differentlyabled" in q or "differently abled" in q:
        return "DIFFERENTLYABLED WELFARE SCHOLARSHIP"
    if "moovalur" in q:
        return "MOOVALUR RAMAMIRTHAM AMMAIYAR"
    if "uzhavar" in q:
        return "UZHAVAR SCHOLARSHIP"
    if "tamil medium" in q:
        return "TAMIL MEDIUM SCHOLARSHIP"
    if "scholarship for research" in q or "research scholarship" in q:
        return "SCHOLARSHIP FOR RESEARCH"
    if "tamilpudhalvan" in q:
        return "TAMILPUDHALVAN SCHEME"
    if "pudhumai penn" in q:
        return "PUDHUMAI PENN SCHEME"
    return ""


def is_endowment_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "endowment scholarship" in q
        or "endowment scholarships" in q
        or q.strip() == "endowment"
    )


def is_university_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q.strip() in {"university scholarship", "university merit scholarship"}


def is_other_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q.strip() == "other scholarship"


def is_scholarship_help_desk_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return q.strip() == "scholarship help desk"


def is_campus_facilities_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "campus facilities and students amenities" in q
        or "campus facilities and student amenities" in q
        or "campus facilities" == q.strip()
        or "students amenities" in q
        or "student amenities" in q
        or "facilities available in the college" in q
        or "what facilities are available in the college" in q
        or "what facilities are available" in q
        or "college facilities" in q
    )


def is_rules_regulations_main_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q.strip() in {
            "rules regulations",
            "rules and regulations",
            "rules regulation",
            "rules and regulation",
        }
        or "rules regulations" in q
        or "rules and regulations" in q
        or "rules regulation" in q
        or "rules and regulation" in q
    )


def is_utility_services_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q.strip() == "utility services"
        or "utility services" in q
    )


def is_important_phone_numbers_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q.strip() == "important phone numbers"
        or "important phone numbers" in q
    )


def is_colleges_affiliated_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "colleges affiliated to m s university" in q
        or "colleges affiliated to m.s university" in q
        or "colleges affiliated to m.s. university" in q
    )


def is_specific_scholarship_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q:
        return False
    # Generic synonym-only queries should return the full scholarships section.
    generic_only_terms = [
        "fee concession",
        "fee concessions",
        "fee waiver",
        "fee waivers",
        "financial aid",
        "financial assistance",
        "fee relief",
        "tuition concession",
        "tuition waiver",
    ]
    if any(term == q for term in generic_only_terms):
        return False
    if is_state_government_scholarship_query(q):
        return True
    if is_national_scholarship_query(q):
        return True
    if is_endowment_scholarship_query(q):
        return True
    if is_university_scholarship_query(q):
        return True
    if is_other_scholarship_query(q):
        return True
    if is_scholarship_help_desk_query(q):
        return True
    if is_national_scholarship_subquery(q):
        return True
    if is_state_scholarship_subquery(q):
        return True
    # Specific scholarship program keywords.
    specific_terms = [
        "university scholarship",
        "university merit scholarship",
        "diocesan scholarship",
        "church workers",
        "non teaching staff",
        "jeeva karunya trust",
        "jeevakarunyatrust",
        "other scholarship",
        "scholarship help desk",
        "post-matric",
        "post matric",
        "merit cum means",
        "minority communities",
        "central sector scheme",
        "ugc",
        "single girl child",
        "rank holders",
    ]
    return any(term in q for term in specific_terms)


SYNONYM_MAP = {
    "fee concession": "concessions and scholarships",
    "fee concessions": "concessions and scholarships",
    "fee waiver": "concessions and scholarships",
    "fee waivers": "concessions and scholarships",
    "financial aid": "concessions and scholarships",
    "financial assistance": "concessions and scholarships",
    "fee relief": "concessions and scholarships",
    "tuition concession": "concessions and scholarships",
    "tuition waiver": "concessions and scholarships",
    "history": "brief history of the college",
    "about college": "brief history of the college",
    "about nmcc": "brief history of the college",
    "administration": "administration",
    "management": "administration",
    "staff": "members of the staff",
    "faculty": "members of the staff",
    "teachers": "members of the staff",
    "non teaching": "non - teaching staff",
    "non-teaching": "non - teaching staff",
    "semester": "curriculum",
    "academic calendar": "nmcc academic calendar 2024 - 2025",
    "calendar": "nmcc academic calendar 2024 - 2025",
    "odd semester": "odd semester",
    "even semester": "even semester",
    "phone number": "important phone numbers",
    "phone numbers": "important phone numbers",
    "contact": "important phone numbers",
    "contacts": "important phone numbers",
    "telephone": "important phone numbers",
    "email": "important website / e-mail / telephone no.",
    "website": "important website / e-mail / telephone no.",
    "websites": "important website / e-mail / telephone no.",
    "fees": "fee, concessions & scholarships",
    "tuition fee": "tuition fee",
    "courses": "courses offered",
    "programs": "courses offered",
    "programmes": "courses offered",
    "curriculum": "curriculum",
    "syllabus": "curriculum",
    "co curricular": "co-curricular activities",
    "co-curricular": "co-curricular activities",
    "extracurricular": "co-curricular activities",
    "activities": "co-curricular activities",
    "sports": "co-curricular activities",
    "ncc": "co-curricular activities",
    "nss": "co-curricular activities",
    "clubs": "associations and clubs",
    "club": "associations and clubs",
    "associations": "associations and clubs",
    "committee": "committees",
    "committees": "committees",
    "cell": "committees",
    "utility": "utility services",
    "utilities": "utility services",
    "services": "utility services",
    "facility": "campus facilities and students' amenities",
    "facilities": "campus facilities and students' amenities",
    "amenities": "campus facilities and students' amenities",
    "campus facilities": "campus facilities and students' amenities",
    "campus law": "campus facilities and students' amenities",
    "campus laws": "campus facilities and students' amenities",
    "hostel": "campus facilities and students' amenities",
    "library": "campus facilities and students' amenities",
    "book bank": "campus facilities and students' amenities",
    "internet": "campus facilities and students' amenities",
    "lab": "campus facilities and students' amenities",
    "labs": "campus facilities and students' amenities",
    "classroom": "campus facilities and students' amenities",
    "classrooms": "campus facilities and students' amenities",
    "rules": "a) general discipline",
    "regulations": "a) general discipline",
    "discipline": "a) general discipline",
    "code of conduct": "code of conduct of the elected students",
    "students council": "students' council",
    "student council": "students' council",
    "union": "students' council",
    "affiliated colleges": "colleges affiliated to m.s. university",
    "colleges affiliated": "colleges affiliated to m.s. university",
    "ms university colleges": "colleges affiliated to m.s. university",
    "rooms": "rooms index",
    "room index": "rooms index",
    "classrooms index": "rooms index",
    "personal record": "personal record",
    "record": "personal record",
    "endowment": "endowment scholarships",
    "endowment scholarships": "endowment scholarships",
    "university scholarship": "university scholarship",
    "university merit scholarship": "university merit scholarship",
    "diocesan scholarship": "diocesan scholarship",
    "other scholarship": "other scholarship",
    "scholarship help desk": "scholarship help desk",
    "student aid fund": "student aid fund",
    "free mid-day meal": "free mid-day meal scheme",
    "good samaritan free education": "good samaritan free education",
}


MAIN_HEADING_SPECS: List[Tuple[str, List[str]]] = [
    ("brief history of the college", ["1. BRIEF HISTORY OF THE COLLEGE", "1. Brief History of the College"]),
    ("administration", ["2. ADMINISTRATION", "2. Administration"]),
    ("members of the staff", ["3. MEMBERS OF THE STAFF", "3. Members of the Staff"]),
    ("courses offered", ["4. COURSES OFFERED", "4. Courses Offered"]),
    ("ph d guides", ["5. PH.D. GUIDES", "5. Ph.D. GUIDES", "5. PH.D GUIDES"]),
    ("fee concessions scholarships", ["6. FEE, CONCESSIONS & SCHOLARSHIPS", "6. Fee, Concessions & Scholarships"]),
    ("campus facilities and students amenities", ["7. CAMPUS FACILITIES AND STUDENTS' AMENITIES", "7. Campus Facilities and Students' Amenities", "7. Campus Facilities and Students’ Amenities"]),
    ("rules and regulations", ["8. RULES AND REGULATIONS", "8. RULES & REGULATIONS", "8. Rules and Regulations", "8. Rules & Regulations"]),
    ("students council", ["9. STUDENTS' COUNCIL", "9. Students' Council", "9. STUDENTS’ COUNCIL"]),
    ("our university", ["10. OUR UNIVERSTIY", "10. OUR UNIVERSITY", "10. Our University"]),
    ("curriculum", ["11. CURRICULUM", "11. Curriculum"]),
    ("co curricular activities", ["11. CO-CURRICULAR ACTIVITIES", "11. Co-Curricular Activities", "12. CO-CURRICULAR ACTIVITIES", "12. Co-Curricular Activities"]),
    ("endowment scholarships", ["13. ENDOWMENT SCHOLARSHIPS", "13. Endowment Scholarships"]),
    ("associations and clubs", ["14. ASSOCIATIONS AND CLUBS", "14. Associations and Clubs", "14. ASSOCIATIONS & CLUBS", "14. Associations & Clubs"]),
    ("committees", ["15. COMMITTEES", "15. Committees"]),
    ("utility services", ["16. UTILITY SERVICES", "16. Utility Services"]),
    ("nmcc academic calendar 2024 2025", ["17. NMCC ACADEMIC CALENDAR 2024 - 2025", "17. NMCC Academic Calendar 2024 - 2025"]),
    ("important phone numbers", ["18. IMPORTANT PHONE NUMBERS", "18. Important Phone Numbers"]),
    ("colleges affiliated to m s university", ["19. COLLEGES AFFILIATED TO M.S UNIVERSITY", "19. COLLEGES AFFILIATED TO M.S. UNIVERSITY"]),
    ("rooms index", ["20. ROOMS INDEX", "20. Rooms Index"]),
]


def extract_query_focus(query: str) -> str:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q:
        return ""
    # If a strong keyword is present, prioritize it.
    for strong in ("curriculum", "semester", "academic calendar", "rules", "regulations"):
        if strong in q:
            return strong
    # Prefer content tokens to strip filler phrases like "give me details of".
    tokens = content_tokens(q)
    if tokens:
        return normalize_heading_key(" ".join(tokens))
    # Fallback: try to capture after common phrases.
    m = re.search(r"(details|information|info)\s+(of|about|on)\s+(.+)$", q)
    if m:
        return normalize_heading_key(m.group(3))
    return q


def expand_query_synonyms(q_norm: str) -> List[str]:
    alts: List[str] = []
    if not q_norm:
        return alts
    # Exact synonym matches.
    if q_norm in SYNONYM_MAP:
        alts.append(SYNONYM_MAP[q_norm])
    # Substring-based synonym matches.
    for key, target in SYNONYM_MAP.items():
        if key in q_norm and target not in alts:
            alts.append(target)
    # Special handling for odd/even semester.
    if "odd semester" in q_norm:
        alts.append("odd semester")
    if "even semester" in q_norm:
        alts.append("even semester")
    return alts


def resolve_main_heading_query(query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm:
        return ""

    candidates = [q_norm]
    for alt in expand_query_synonyms(q_norm):
        alt_norm = normalize_heading_key(alt)
        if alt_norm and alt_norm not in candidates:
            candidates.append(alt_norm)

    best_key = ""
    best_score = 0.0
    for cand in candidates:
        cand_tokens = set(content_tokens(cand))
        for key, _ in MAIN_HEADING_SPECS:
            key_norm = normalize_heading_key(key)
            key_tokens = set(content_tokens(key_norm))
            score = 0.0

            if cand == key_norm:
                score = 3.0
            elif cand and (cand in key_norm or key_norm in cand):
                score = 2.0

            if cand_tokens and key_tokens:
                overlap = len(cand_tokens & key_tokens) / max(1, len(cand_tokens))
                score += overlap
                if cand_tokens == key_tokens:
                    score += 0.4

            if score > best_score:
                best_score = score
                best_key = key_norm

    if best_score < 0.85:
        return ""
    return best_key


def extract_main_heading_block(source_text: str, query: str) -> str:
    key = resolve_main_heading_query(query)
    if not key:
        return ""

    for i, (spec_key, start_markers) in enumerate(MAIN_HEADING_SPECS):
        if normalize_heading_key(spec_key) != key:
            continue

        end_markers: List[str] = []
        if i + 1 < len(MAIN_HEADING_SPECS):
            end_markers = MAIN_HEADING_SPECS[i + 1][1]

        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=start_markers,
            end_markers=end_markers,
        )
        if block:
            return strip_leading_heading_line(block)
        return ""

    return ""


def is_staff_council_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "staff council" in q
        or q.strip() == "council staff"
    )


def is_staff_council_members_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "staff council members" in q
        or "staff council members 2024 25" in q
    )


def is_university_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "our university" in q
        or "manonmaniam sundaranar university" in q
        or "ms university" in q
        or "vice chancellor" in q
        or "registrar" in q
        or "controller of examinations" in q
        or "centre for research" in q
    )


def is_non_teaching_staff_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "non teaching staff" in q
        or "non-teaching staff" in normalize_mojibake(query).lower()
        or q.strip() == "non teaching"
    )


def is_history_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        "history of nmcc" in q
        or "history of college" in q
        or "brief history of the college" in q
        or q.strip() == "history"
        or q.strip() == "brief history"
    )


def is_about_nmcc_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q in {"college", "about nmcc", "about the nmcc", "about college", "about the college"}
        or "tell me about nmcc" in q
        or "tell me about college" in q
        or "tell me about the college" in q
        or "tell me about nmcc" in q
        or "details of nmcc" in q
        or "nmcc details" in q
        or "details of college" in q
        or "college details" in q
        or "details about college" in q
        or "details of the college" in q
        or "details about nmcc" in q
        or "about nmcc college" in q
        or "about nesamony memorial christian college" in q
        or "about nesamony college" in q
    )


def trim_history_tail(block: str) -> str:
    if not block:
        return ""
    out = normalize_mojibake(block)
    lines = [normalize_text(x) for x in out.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    cut_patterns = [
        r"^EMBLEM$",
        r"^Vision$",
        r"^Mission$",
        r"^SUCCESSION\s*-\s*LIST OF SECRETARIES$",
        r"^SUCCESSION\s*-\s*LIST OF PRINCIPALS$",
        r"^2\.\s+ADMINISTRATION(?: AND MANAGEMENT)?$",
    ]

    cut_at = len(lines)
    for i, line in enumerate(lines):
        if any(re.match(pattern, line, flags=re.I) for pattern in cut_patterns):
            cut_at = i
            break

    return "\n".join(lines[:cut_at]).strip()


def is_contents_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q:
        return False

    if q == "contents" or q == "content":
        return True
    if "table of contents" in q or q == "index":
        return True

    # Match common natural-language forms:
    # "what are the contents", "tell me about the contents",
    # "contents are there", "content list", "index".
    content_terms = ["contents", "content", "table of contents", "index"]
    intent_terms = [
        "show",
        "display",
        "tell",
        "about",
        "what",
        "list",
        "give",
        "provide",
        "there",
        "available",
    ]
    has_content_term = any(term in q for term in content_terms)
    has_intent_term = any(term in q for term in intent_terms)
    return has_content_term and has_intent_term


def is_unknown_or_gibberish_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    raw_tokens = tokenize(q)
    if not raw_tokens:
        return True

    # Natural-language queries like "tell me about college" can lose all
    # content tokens after stop-word removal; they should not be treated as gibberish.
    if len(raw_tokens) >= 3 and any(re.search(r"[aeiou]", t) for t in raw_tokens):
        return False

    toks = content_tokens(q)
    if not toks:
        return True

    if any(t in ALLOWED_ACRONYMS for t in toks):
        return False

    # If all meaningful tokens are vowel-less keyboard noise (e.g., "hjkl"),
    # treat it as unknown input and avoid returning unrelated sections.
    has_vowel_token = any(re.search(r"[aeiou]", t) for t in toks)
    return not has_vowel_token


def is_greeting_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    if not q:
        return False
    if q in {"good morning", "good afternoon", "good evening"}:
        return True
    if q.startswith(("hi ", "hello ", "hey ", "hai ")):
        return True

    # Handle stretched/variant greetings: "haiii", "heyyyy", "hiiiiii", etc.
    token = q.split()[0]
    return bool(
        re.fullmatch(r"h+i+", token)
        or re.fullmatch(r"ha+i+", token)
        or re.fullmatch(r"he+y+", token)
        or re.fullmatch(r"hello+", token)
        or token in {"hlo", "helo", "helloo", "hii", "haii"}
    )


def is_bye_query(query: str) -> bool:
    q = normalize_heading_key(normalize_mojibake(query))
    return bool(
        q in {"bye", "good bye", "goodbye", "see you", "see ya", "thanks bye", "thank you bye"}
        or q.startswith("bye ")
    )


def extract_associations_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    sec = find_section_by_heading_keyword(sections, "associations and clubs")
    if sec:
        return sec

    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["14. ASSOCIATIONS AND CLUBS", "14. Associations and Clubs"],
        end_markers=["15. COMMITTEES", "15. Committees"],
    )


def extract_scholarship_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    # Primary: extract directly from full source to ensure the whole scholarship
    # content is returned (including Student Aid Fund and Free Mid-Day Meal Scheme).
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "CONCESSIONS AND SCHOLARSHIPS",
            "Concessions and Scholarships",
        ],
        end_markers=[
            "7. CAMPUS FACILITIES AND STUDENTS' AMENITIES",
            "7. CAMPUS FACILITIES AND STUDENTSâ€™ AMENITIES",
            "7. Campus Facilities and Students’ Amenities",
        ],
    )
    if block:
        return block

    # Fallback: extract from section split if raw marker search fails.
    fees_section = find_section_by_heading_keyword(sections, "fee, concessions & scholarships")
    if not fees_section:
        return ""
    return extract_block_between_markers(
        source_text=fees_section,
        start_markers=[
            "CONCESSIONS AND SCHOLARSHIPS",
            "Concessions and Scholarships",
        ],
        end_markers=[],
    )


def extract_concessions_and_scholarships_intro_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "CONCESSIONS AND SCHOLARSHIPS",
            "Concessions and Scholarships",
        ],
        end_markers=[
            "STATE GOVERNMENT SCHOLARSHIPS",
        ],
    )
    return strip_leading_heading_line(block) if block else ""


def extract_state_government_scholarships_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["STATE GOVERNMENT SCHOLARSHIPS"],
        end_markers=["NATIONAL SCHOLARSHIPS", "National Scholarships"],
    )


def extract_national_scholarships_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=["NATIONAL SCHOLARSHIPS", "National Scholarships"],
        end_markers=[
            "UNIVERSITY SCHOLARSHIP",
            "UNIVERSITY MERIT SCHOLARSHIP",
            "SCHOLARSHIP HELP DESK",
        ],
    )


def extract_ugc_scholarships_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["SCHOLARSHIPS BY UNIVERSITY GRANTS COMMISSION"],
        end_markers=[
            "UNIVERSITY SCHOLARSHIP",
            "UNIVERSITY MERIT SCHOLARSHIP",
            "SCHOLARSHIP HELP DESK",
        ],
    )
    if not block:
        return ""
    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if lines and "scholarships by university grants commission" in normalize_heading_key(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_ugc_scholarship_subsection(source_text: str, key: str) -> str:
    key_norm = normalize_heading_key(key)
    if key_norm == "post graduate scholarship for single girl child":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "a) POST GRADUATE SCHOLARSHIP FOR SINGLE GIRL CHILD",
                "POST GRADUATE SCHOLARSHIP FOR SINGLE GIRL CHILD",
            ],
            end_markers=[
                "b) POST GRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS",
                "POSTGRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            if lines and "post graduate scholarship for single girl child" in normalize_heading_key(lines[0]):
                lines = lines[1:]
            return "\n".join(lines).strip()
    if key_norm == "post graduate scholarship for university rank holders":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "b) POST GRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS",
                "POSTGRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS",
                "POST GRADUATE SCHOLARSHIP FOR UNIVERSITY RANK HOLDERS",
            ],
            end_markers=[
                "UNIVERSITY SCHOLARSHIP",
                "UNIVERSITY MERIT SCHOLARSHIP",
                "SCHOLARSHIP HELP DESK",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            if lines and "post graduate scholarship for university rank holders" in normalize_heading_key(lines[0]):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return ""


def extract_university_scholarship_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["UNIVERSITY MERIT SCHOLARSHIP"],
        end_markers=[
            "DIOCESAN SCHOLARSHIP",
            "CHURCH WORKERS AND NON-TEACHING STAFF CHILDREN'S SCHOLARSHIP",
            "OTHER SCHOLARSHIP",
        ],
    )
    if not block:
        return ""
    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if lines and normalize_heading_key(lines[0]) == "university merit scholarship":
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_scholarship_help_desk_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["SCHOLARSHIP HELP DESK"],
        end_markers=["STUDENT AID FUND"],
    )
    if not block:
        return ""
    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if lines and normalize_heading_key(lines[0]) == "scholarship help desk":
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_campus_facilities_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "7. CAMPUS FACILITIES AND STUDENTS' AMENITIES",
            "7. CAMPUS FACILITIES AND STUDENTSâ€™ AMENITIES",
            "7. CAMPUS FACILITIES AND STUDENTS’ AMENITIES",
        ],
        end_markers=[
            "8. RULES AND REGULATIONS",
            "8. RULES & REGULATIONS",
            "8. Rules and Regulations",
            "8. Rules & Regulations",
        ],
    )
    if not block:
        return ""
    return strip_leading_heading_line(block)


def extract_rules_regulations_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "8. RULES AND REGULATIONS",
            "8. RULES & REGULATIONS",
            "8. Rules and Regulations",
            "8. Rules & Regulations",
        ],
        end_markers=[
            "9. STUDENTS' COUNCIL",
            "9. STUDENTSâ€™ COUNCIL",
            "9. Students' Council",
            "10. OUR UNIVERSTIY",
            "10. OUR UNIVERSITY",
        ],
    )
    if not block:
        return ""
    return strip_leading_heading_line(block)


def extract_utility_services_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "16. UTILITY SERVICES",
            "16. Utility Services",
        ],
        end_markers=[
            "17. NMCC ACADEMIC CALENDAR 2024 - 2025",
            "17. NMCC Academic Calendar 2024 - 2025",
        ],
    )
    if not block:
        return ""
    return strip_leading_heading_line(block)


def extract_important_phone_numbers_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "18. IMPORTANT PHONE NUMBERS",
            "18. Important Phone Numbers",
        ],
        end_markers=[
            "19. COLLEGES AFFILIATED TO M.S UNIVERSITY",
            "19. COLLEGES AFFILIATED TO M.S. UNIVERSITY",
        ],
    )
    if not block:
        return ""
    return strip_leading_heading_line(block)


def extract_colleges_affiliated_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "19. COLLEGES AFFILIATED TO M.S UNIVERSITY",
            "19. COLLEGES AFFILIATED TO M.S. UNIVERSITY",
        ],
        end_markers=[
            "20. ROOMS INDEX",
            "20. Rooms Index",
        ],
    )
    if not block:
        return ""
    return strip_leading_heading_line(block)


def extract_state_scholarship_subsection(source_text: str, key: str) -> str:
    # Hard extract for Differently Abled to avoid leakage from other sections.
    if normalize_heading_key(key) in {"differentlyabled welfare scholarship", "differently abled welfare scholarship"}:
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "c) DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "c) DIFFERENTLYABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLY ABLED WELFARE SCHOLARSHIP",
                "DIFFERENTLYABLED WELFARE SCHOLARSHIP",
            ],
            end_markers=[
                "d) MOOVALUR RAMAMIRTHAM AMMAIYAR",
                "d) MOOVALUR RAMAMIRTHAM AMMAIYAR HIGHER EDUCATION",
                "MOOVALUR RAMAMIRTHAM AMMAIYAR",
            ],
        )
        if block:
            return block

        # Line-based fallback: capture until the next lettered heading.
        lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
        lines = [x for x in lines if x and not x.startswith("--- Page")]
        start_i = -1
        for i, line in enumerate(lines):
            if normalize_heading_key(line) in {
                "c differently abled welfare scholarship",
                "c differentlyabled welfare scholarship",
                "differently abled welfare scholarship",
                "differentlyabled welfare scholarship",
            }:
                start_i = i
                break
        if start_i < 0:
            return ""
        end_i = len(lines)
        for i in range(start_i + 1, len(lines)):
            if re.match(r"^[d-h]\)\s+", lines[i], flags=re.I):
                end_i = i
                break
        return "\n".join(lines[start_i:end_i]).strip()
    if normalize_heading_key(key) == "tamilpudhalvan scheme":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "h) TAMILPUDHALVAN SCHEME",
                "h) TAMIL PUDHALVAN SCHEME",
                "TAMILPUDHALVAN SCHEME",
                "TAMIL PUDHALVAN SCHEME",
            ],
            end_markers=[
                "I) PUDHUMAI PENN SCHEME",
                "i) PUDHUMAI PENN SCHEME",
                "PUDHUMAI PENN SCHEME",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            if lines and normalize_heading_key(lines[0]) in {
                "h tamilpudhalvan scheme",
                "h tamil pudhalvan scheme",
                "tamilpudhalvan scheme",
                "tamil pudhalvan scheme",
            }:
                lines = lines[1:]
            if lines and normalize_heading_key(lines[0]) == "eligibility criteria":
                return "\n".join(lines[:2]).strip()
            return "\n".join(lines).strip()
    if normalize_heading_key(key) == "pudhumai penn scheme":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "I) PUDHUMAI PENN SCHEME",
                "i) PUDHUMAI PENN SCHEME",
                "PUDHUMAI PENN SCHEME",
            ],
            end_markers=[
                "NATIONAL SCHOLARSHIPS",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            if lines and "pudhumai penn scheme" in normalize_heading_key(lines[0]):
                lines = lines[1:]
            return "\n".join(lines).strip()
    block = extract_state_government_scholarships_block(source_text)
    if not block:
        block = source_text

    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    key_norm = normalize_heading_key(key)
    start_i = -1
    for i, line in enumerate(lines):
        line_norm = normalize_heading_key(line)
        if key_norm in line_norm:
            start_i = i
            break
    if start_i < 0:
        return ""

    end_i = len(lines)
    for i in range(start_i + 1, len(lines)):
        if re.match(r"^[a-h]\)\s+", lines[i], flags=re.I):
            end_i = i
            break
        if "national scholarships" in normalize_heading_key(lines[i]):
            end_i = i
            break

    out = lines[start_i:end_i]
    if out and key_norm in normalize_heading_key(out[0]):
        out = out[1:]
    if out and re.fullmatch(r"\d{1,3}", out[0].strip()):
        out = out[1:]
    return "\n".join(out).strip()


def extract_national_scholarship_subsection(source_text: str, key: str) -> str:
    key_norm = normalize_heading_key(key)
    if key_norm == "central sector scheme of scholarship for students with disabilities":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "d) CENTRAL SECTOR SCHEME OF SCHOLARSHIP FOR STUDENTS",
                "CENTRAL SECTOR SCHEME OF SCHOLARSHIP FOR STUDENTS",
            ],
            end_markers=[
                "e) NATIONAL FELLOWSHIP & SCHOLARSHIP FOR HIGHER",
                "NATIONAL FELLOWSHIP & SCHOLARSHIP FOR HIGHER",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            while lines and (
                "central sector scheme of scholarship for students" in normalize_heading_key(lines[0])
                or normalize_heading_key(lines[0]) == "with disabilities"
            ):
                lines = lines[1:]
            while lines:
                tail_norm = normalize_heading_key(lines[-1])
                if (
                    "national fellowship scholarship for higher" in tail_norm
                    or tail_norm.startswith("e national fellowship scholarship for higher")
                    or tail_norm.startswith("e national fellowship")
                    or re.match(r"^e\b", tail_norm)
                ):
                    lines = lines[:-1]
                    continue
                break
            out = "\n".join(lines).strip()
            out = re.sub(
                r"\n\s*e\)\s*NATIONAL FELLOWSHIP\s*&\s*SCHOLARSHIP\s+FOR\s+HIGHER.*$",
                "",
                out,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            return out
    if key_norm == "national fellowship scholarship for higher education of scheduled tribe students":
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "e) NATIONAL FELLOWSHIP & SCHOLARSHIP FOR HIGHER",
                "NATIONAL FELLOWSHIP & SCHOLARSHIP FOR HIGHER",
            ],
            end_markers=[
                "SCHOLARSHIPS BY UNIVERSITY GRANTS COMMISSION",
            ],
        )
        if block:
            lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
            lines = [x for x in lines if x and not x.startswith("--- Page")]
            while lines and (
                "national fellowship scholarship for higher" in normalize_heading_key(lines[0])
                or normalize_heading_key(lines[0]) == "education of scheduled tribe students"
            ):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return ""


def extract_other_scholarship_subsection(source_text: str, query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm:
        return ""

    if is_other_scholarship_query(q_norm):
        block = extract_block_between_markers(
            source_text=source_text,
            start_markers=[
                "OTHER SCHOLARSHIP",
            ],
            end_markers=[
                "SCHOLARSHIP HELP DESK",
                "STUDENT AID FUND",
            ],
        )
        if not block:
            return ""
        lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
        lines = [x for x in lines if x and not x.startswith("--- Page")]
        if lines and normalize_heading_key(lines[0]) == "other scholarship":
            lines = lines[1:]
        return "\n".join(lines).strip()

    if not (
        "indian jeeva karunya trust scholarship" in q_norm
        or "indian jeevakarunyatrust scholarship" in q_norm
        or "jeeva karunya trust scholarship" in q_norm
        or "jeevakarunyatrust scholarship" in q_norm
    ):
        return ""

    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "SCHOLARSHIP HELP DESK",
        ],
        end_markers=[
            "STUDENT AID FUND",
        ],
    )
    if not block:
        return ""
    lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]
    if lines and normalize_heading_key(lines[0]) == "scholarship help desk":
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_endowment_scholarship_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    sec = find_section_by_heading_keyword(sections, "endowment scholarships")
    if sec:
        return sec

    return extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "13. ENDOWMENT SCHOLARSHIPS",
            "13. Endowment Scholarships",
            "ENDOWMENT SCHOLARSHIPS",
        ],
        end_markers=[
            "14. ASSOCIATIONS AND CLUBS",
            "14. Associations and Clubs",
            "14. ASSOCIATIONS & CLUBS",
            "14. Associations & Clubs",
        ],
    )


def extract_tuition_fees_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    fees_section = find_section_by_heading_keyword(sections, "fee, concessions & scholarships")
    if not fees_section:
        fees_section = source_text

    block = extract_block_between_markers(
        source_text=fees_section,
        start_markers=["TUITION FEE", "Tuition Fee"],
        end_markers=[
            "CONCESSIONS AND SCHOLARSHIPS",
            "Concessions and Scholarships",
        ],
    )
    if not block:
        return ""

    # Return only content lines, not the heading label.
    lines = [normalize_text(x) for x in block.splitlines()]
    lines = [x for x in lines if x]
    if lines and normalize_heading_key(lines[0]) == "tuition fee":
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_students_council_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    council_section = find_section_by_heading_keyword(sections, "students council")
    if council_section:
        block = extract_block_between_markers(
            source_text=council_section,
            start_markers=[
                "Students' Council Constitution",
                "Students’ Council Constitution",
                "ARTICLE-1: Aim",
            ],
            end_markers=[],
        )
        return block if block else council_section

    return extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "9. STUDENTS' COUNCIL",
            "9. STUDENTS’ COUNCIL",
            "Students' Council Constitution",
            "Students’ Council Constitution",
        ],
        end_markers=[
            "10. OUR UNIVERSTIY",
            "10. OUR UNIVERSITY",
            "10. Our University",
        ],
    )


def extract_staff_council_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["THE STAFF COUNCIL"],
        end_markers=["STAFF COUNCIL MEMBERS 2024-25"],
    )
    return strip_leading_heading_line(block) if block else ""


def extract_staff_council_members_block(source_text: str) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=["STAFF COUNCIL MEMBERS 2024-25"],
        end_markers=["3. MEMBERS OF THE STAFF"],
    )
    return strip_leading_heading_line(block) if block else ""


def extract_university_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    sec = find_section_by_heading_keyword(sections, "our university")
    if sec:
        return sec

    return extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "10. OUR UNIVERSTIY",
            "10. OUR UNIVERSITY",
            "10. Our University",
        ],
        end_markers=[
            "11. CO-CURRICULAR ACTIVITIES",
            "11. CO CURRICULAR ACTIVITIES",
            "11. Co-Curricular Activities",
            "11. Co-curricular Activities",
        ],
    )


def extract_non_teaching_staff_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "NON - TEACHING STAFF",
            "NON-TEACHING STAFF",
            "NON TEACHING STAFF",
        ],
        end_markers=[
            "4. COURSES OFFERED",
            "4. Courses Offered",
            "5. PH.D. GUIDES",
            "5. Ph.D. GUIDES",
        ],
    )
    if block:
        return strip_leading_heading_line(block)

    staff_section = find_section_by_heading_keyword(sections, "members of staff")
    if staff_section:
        block = extract_block_between_markers(
            source_text=staff_section,
            start_markers=[
                "NON - TEACHING STAFF",
                "NON-TEACHING STAFF",
                "NON TEACHING STAFF",
            ],
            end_markers=[],
        )
        if block:
            return strip_leading_heading_line(block)
    return ""


def extract_teaching_staff_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "TEACHING STAFF",
        ],
        end_markers=[
            "NON - TEACHING STAFF",
            "NON-TEACHING STAFF",
            "NON TEACHING STAFF",
            "4. COURSES OFFERED",
            "4. Courses Offered",
        ],
    )
    if block:
        return block

    staff_section = find_section_by_heading_keyword(sections, "members of staff")
    if staff_section:
        block = extract_block_between_markers(
            source_text=staff_section,
            start_markers=["TEACHING STAFF"],
            end_markers=[
                "NON - TEACHING STAFF",
                "NON-TEACHING STAFF",
                "NON TEACHING STAFF",
            ],
        )
        return block if block else staff_section
    return ""


def extract_members_of_staff_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    sec = find_section_by_heading_keyword(sections, "members of the staff")
    if sec:
        return sec

    return extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "3. MEMBERS OF THE STAFF",
            "3. Members of the Staff",
            "MEMBERS OF THE STAFF",
        ],
        end_markers=[
            "4. COURSES OFFERED",
            "4. Courses Offered",
            "4. COURSES",
        ],
    )


def extract_history_block(source_text: str, sections: List[Tuple[str, str]]) -> str:
    # Prefer the full handbook history narrative that starts before the
    # numbered heading and stops before EMBLEM.
    full_block = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "Nesamony Memorial Christian College is a prestigious",
        ],
        end_markers=[
            "EMBLEM",
            "2. ADMINISTRATION",
            "2. Administration",
            "2. ADMINISTRATION AND MANAGEMENT",
        ],
    )
    if full_block:
        return trim_history_tail(full_block)

    sec = find_section_by_heading_keyword(sections, "brief history of the college")
    if sec and len(sec.splitlines()) > 4:
        return trim_history_tail(sec)

    fallback = extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "1. BRIEF HISTORY OF THE COLLEGE",
            "1. Brief History of the College",
            "BRIEF HISTORY OF THE COLLEGE",
        ],
        end_markers=[
            "EMBLEM",
            "2. ADMINISTRATION",
            "2. Administration",
            "2. ADMINISTRATION AND MANAGEMENT",
        ],
    )
    return trim_history_tail(fallback)


def extract_about_nmcc_block(source_text: str) -> str:
    return extract_block_between_markers(
        source_text=source_text,
        start_markers=[
            "Nesamony Memorial Christian College is a prestigious",
            "NESAMONY MEMORIAL CHRISTIAN COLLEGE IS A PRESTIGIOUS",
        ],
        end_markers=[
            "2. ADMINISTRATION",
            "2. Administration",
        ],
    )


def extract_contents_block(source_text: str) -> str:
    lines: List[str] = []
    for raw in source_text.splitlines():
        line = normalize_text(normalize_mojibake(raw))
        if re.match(r"^\s*\d+\.\s+.+?\s+\.{2,}\s*\d+\s*$", line):
            lines.append(line)
    return "\n".join(lines).strip()


def build_subheading_sections(source_text: str, sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    for heading, block in sections:
        main_title = normalize_heading_key(re.sub(r"^\d+\.\s*", "", heading))

        lines = [normalize_text(normalize_mojibake(x)) for x in block.splitlines()]
        lines = [x for x in lines if x and not x.startswith("--- Page")]

        starts: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            line_norm = normalize_text(line)
            title = ""

            # Numbered short sub-headings.
            m = re.match(r"^(\d{1,3})\.\s+([A-Za-z][A-Za-z0-9 '&()/.\-]{2,})$", line_norm)
            if m:
                cand = normalize_text(m.group(2))
                cand_l = cand.lower()
                if (
                    normalize_heading_key(cand) != main_title
                    and len(cand.split()) <= 12
                    and not re.search(r"\bdr\.?\b", cand.lower())
                    and cand_l not in {"vacant", "guest faculty"}
                    and not cand_l.startswith("vacant ")
                    and not cand_l.startswith("guest faculty")
                ):
                    title = f"{m.group(1)}. {cand}"
            else:
                m = re.match(r"^([A-Ha-h])\)\s+([A-Za-z][A-Za-z0-9 '&()/.\-]{2,})$", line_norm)
                if m:
                    cand = normalize_text(m.group(2))
                    if normalize_heading_key(cand) != main_title:
                        title = f"{m.group(1).lower()}) {cand}"
                # Non-numbered uppercase/department sub-headings like BOTANY FACULTY.
                elif re.match(r"^[A-Z][A-Z0-9 '&()/.\-]{3,}$", line_norm) and len(line_norm.split()) <= 8:
                    if normalize_heading_key(line_norm) != main_title:
                        title = line_norm

            if title:
                starts.append((i, title))

        for idx, (start_i, subhead) in enumerate(starts):
            end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            sub_block = "\n".join(lines[start_i:end_i]).strip()
            if sub_block:
                out.append((subhead, sub_block))
    return out


def match_subheading_section(query: str, sub_sections: List[Tuple[str, str]]) -> str:
    if not sub_sections:
        return ""

    q_norm = normalize_heading_key(query)
    q_tokens = set(content_tokens(q_norm))
    best_score = 0.0
    best_block = ""

    for heading, block in sub_sections:
        h_norm = normalize_heading_key(re.sub(r"^\d+\.\s*", "", heading))
        h_tokens = set(content_tokens(h_norm))

        overlap = 0.0
        if q_tokens and h_tokens:
            overlap = len(q_tokens & h_tokens) / max(1, len(q_tokens))

        sim = SequenceMatcher(None, q_norm, h_norm).ratio()
        score = (0.7 * overlap) + (0.3 * sim)

        if q_norm and (q_norm == h_norm or q_norm in h_norm or h_norm in q_norm):
            score += 0.35

        if q_tokens and h_tokens and q_tokens.issubset(h_tokens):
            score += 0.25

        # Prefer faculty-style headings for faculty queries.
        if ("faculty" in q_norm) and ("faculty" in h_norm):
            score += 0.25

        if score > best_score:
            best_score = score
            best_block = block

    if best_score < 0.40:
        return ""
    return best_block


def find_subheading_block_in_text(source_text: str, query: str) -> str:
    q_norm = normalize_heading_key(normalize_mojibake(query))
    if not q_norm:
        return ""

    lines = [normalize_text(normalize_mojibake(x)) for x in source_text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("--- Page")]

    q_tokens = set(content_tokens(q_norm))
    best_i = -1
    best_score = 0.0

    for i, line in enumerate(lines):
        if is_toc_line(line):
            continue
        if not (is_lettered_heading(line) or is_numbered_short_heading(line) or is_upper_heading(line)):
            continue
        l_norm = normalize_heading_key(line)
        if not l_norm:
            continue

        score = 0.0
        if q_norm == l_norm:
            score = 3.0
        elif q_norm in l_norm or l_norm in q_norm:
            score = 2.0

        if q_tokens:
            l_tokens = set(content_tokens(l_norm))
            if l_tokens:
                overlap = len(q_tokens & l_tokens) / max(1, len(q_tokens))
                score += overlap
                if q_tokens.issubset(l_tokens):
                    score += 0.3

        if score > best_score:
            best_score = score
            best_i = i

    if best_score < 0.45 or best_i < 0:
        return ""

    end_i = len(lines)
    for j in range(best_i + 1, len(lines)):
        if is_any_heading_line(lines[j]) and not is_toc_line(lines[j]):
            end_i = j
            break

    block_lines = lines[best_i:end_i]
    block = "\n".join(block_lines).strip()

    # If the matched heading is a parent label and has no body before the
    # next heading, try returning the immediate child heading block.
    if end_i < len(lines):
        non_heading = [ln for ln in block_lines[1:] if not is_any_heading_line(ln)]
        if not non_heading:
            parent_norm = normalize_heading_key(block_lines[0])
            child_line = lines[end_i]
            child_norm = normalize_heading_key(child_line)
            parent_tokens = set(content_tokens(parent_norm))
            child_tokens = set(content_tokens(child_norm))
            if parent_tokens and parent_tokens.issubset(child_tokens):
                # Recompute block for the child heading.
                next_end = len(lines)
                for k in range(end_i + 1, len(lines)):
                    if is_any_heading_line(lines[k]) and not is_toc_line(lines[k]):
                        next_end = k
                        break
                block = "\n".join(lines[end_i:next_end]).strip()

    return strip_leading_heading_line(block)


def load_data() -> List[str]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [normalize_text(str(c)) for c in json.load(f)]
    return [c for c in chunks if c]


# ---------- Chat Loop ----------
def chat():
    chunks = load_data()
    intents = load_intents()
    source_text = load_source_text(chunks)
    heading_sections = build_heading_sections(source_text)
    subheading_sections = build_subheading_sections(source_text, heading_sections)
    heading_index = build_heading_index_from_text(source_text)

    print("\nPDF Chatbot Ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        answer = answer_query(
            query=query,
            source_text=source_text,
            heading_sections=heading_sections,
            subheading_sections=subheading_sections,
            heading_index=heading_index,
            intents=intents,
        )

        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    chat()
