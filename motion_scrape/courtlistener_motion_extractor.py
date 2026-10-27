"""
CourtListener API v4 Motion Full-Text & Outcome Extractor (2016–Present, JSON Output)

What this script does:
1. Searches CourtListener API v4 (`/api/rest/v4/search/`) across Illinois federal
   district courts (`ilnd`, `ilcd`, `ilsd`) for specific motion types filed between
   2016-01-01 and today.
2. Filters for filings that have archived PDFs/text in the free RECAP Archive
   (`available_only="on"`).
3. Fetches the full extracted/OCR text (`plain_text`) for each motion from
   `/api/rest/v4/recap-documents/{doc_id}/`.
4. Determines whether each motion Passed ("pass" = granted), Failed ("fail" = denied),
   was Partially Granted ("partial" = granted in part / denied in part), or is
   "pending_or_unknown" by:
   - Checking the docket entry description and document text for direct dispositions, and
   - Searching the same docket (`docket_id`) for subsequent orders or minute entries
     that rule on the motion (`granted` vs. `denied`).
5. Saves all records—including `plain_text`, `outcome_label`, `ruling_evidence`,
   and `pdf_url`—directly into `illinois_motions_2016_present.json`.
"""

import json
import os
import re
import time
from datetime import date
import requests

try:
    from google.colab import userdata, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ==============================================================================
# STEP 1: Connect to CourtListener & Configure Settings
# ==============================================================================

# Load token securely from Colab Secrets or environment variable.
# IMPORTANT: Reset any previously exposed token at:
# https://www.courtlistener.com/profile/api-token/
API_TOKEN = "02482887a6c8a11b33960588850d5651ccb466b4"
if IN_COLAB:
    try:
        API_TOKEN = userdata.get("COURTLISTENER_API_TOKEN")
    except Exception:
        API_TOKEN = None

if not API_TOKEN:
    API_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN", "")

if not API_TOKEN:
    raise ValueError(
        "API token not found. Set 'COURTLISTENER_API_TOKEN' in Colab Secrets "
        "or as an environment variable."
    )

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
RECAP_DOC_URL = "https://www.courtlistener.com/api/rest/v4/recap-documents/"
STORAGE_BASE_URL = "https://storage.courtlistener.com/"

HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "Accept": "application/json",
}

# Date range: 2016-01-01 to today
START_DATE = "2016-01-01"
END_DATE = date.today().isoformat()

# Rate-limit & extraction settings:
# Default free CourtListener accounts allow 5 requests/min, 50/hour, 125/day.
# Keep MAX_DOCS_PER_COMBO modest on a free tier, or raise it on an EDU/Member tier.
MAX_DOCS_PER_COMBO = 3
POLITE_DELAY_SECONDS = 12  # ~5 requests/min safe pacing for API endpoints
CHECK_DOCKET_FOR_RULING = True  # Queries the docket for subsequent grant/deny orders
DOWNLOAD_PDF_FILES = False      # Set True if you also want raw .pdf files saved locally

JSON_FILENAME = "illinois_motions_2016_present.json"
PDF_DIR = os.path.join("extracted_motions", "pdfs")
if DOWNLOAD_PDF_FILES:
    os.makedirs(PDF_DIR, exist_ok=True)


# ==============================================================================
# STEP 2: Define Courts and Motion Types
# ==============================================================================

courts = {
    "ilnd": "Northern District of Illinois",
    "ilcd": "Central District of Illinois",
    "ilsd": "Southern District of Illinois",
}

motions = [
    "summary judgment",
    "motion to dismiss",
    "motion to compel",
    "preliminary injunction",
]

print(f"Date range: {START_DATE} to {END_DATE}")
print("Total court/motion combinations:", len(courts) * len(motions))


# ==============================================================================
# STEP 3: Helper Functions for API Requests, Full Text & Pass/Fail Labeling
# ==============================================================================

def request_with_retry(url, headers=None, params=None, stream=False, max_attempts=3):
    """Executes an HTTP GET request with automatic 429 Retry-After handling."""
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(10, 90),
                stream=stream,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                try:
                    wait = max(1, int(retry_after))
                except ValueError:
                    wait = 60
                print(f"  [429 Rate Limit] Waiting {wait + 2} seconds...")
                time.sleep(wait + 2)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            print(f"  [Timeout] Attempt {attempt + 1}/{max_attempts} for {url}")
        except requests.exceptions.RequestException as error:
            print(f"  [Request Error] {error}")
            break

    return None


def fetch_recap_document_text(doc_id):
    """
    Fetches full extracted text (`plain_text`) and file metadata for a single
    RECAP Document ID from `/api/rest/v4/recap-documents/{doc_id}/`.
    """
    url = f"{RECAP_DOC_URL}{doc_id}/"
    params = {
        "fields": (
            "id,plain_text,filepath_local,is_available,"
            "ocr_status,page_count,description"
        )
    }
    time.sleep(POLITE_DELAY_SECONDS)
    resp = request_with_retry(url, headers=HEADERS, params=params)
    if resp is None:
        return {}
    return resp.json()


# Regex patterns for classifying motion outcomes from docket descriptions or orders
PARTIAL_PATTERN = re.compile(
    r"\b(?:granted\s+in\s+part\s+and\s+denied\s+in\s+part|"
    r"denied\s+in\s+part\s+and\s+granted\s+in\s+part|"
    r"granted\s+in\s+part)\b",
    re.IGNORECASE,
)
GRANTED_PATTERN = re.compile(
    r"\b(?:motion[^.;\n]{0,80}\bis\s+granted|"
    r"grants\s+(?:plaintiff'?s?|defendant'?s?|the)?\s*motion|"
    r"motion\s+for\s+[^.;\n]{0,50}\s+is\s+granted|"
    r"motion\s+to\s+[^.;\n]{0,50}\s+is\s+granted|"
    r"\bgranted\b)",
    re.IGNORECASE,
)
DENIED_PATTERN = re.compile(
    r"\b(?:motion[^.;\n]{0,80}\bis\s+denied|"
    r"denies\s+(?:plaintiff'?s?|defendant'?s?|the)?\s*motion|"
    r"motion\s+for\s+[^.;\n]{0,50}\s+is\s+denied|"
    r"motion\s+to\s+[^.;\n]{0,50}\s+is\s+denied|"
    r"denied\s+as\s+moot|"
    r"\bdenied\b|\bstricken\b)",
    re.IGNORECASE,
)


def classify_ruling_text(text):
    """
    Classifies a ruling description or order snippet into:
      - ("partial", "granted_in_part")
      - ("pass", "granted")
      - ("fail", "denied")
      - (None, None) if no clear ruling language is present
    """
    if not text:
        return None, None

    if PARTIAL_PATTERN.search(text):
        return "partial", "granted_in_part"

    has_grant = bool(GRANTED_PATTERN.search(text))
    has_deny = bool(DENIED_PATTERN.search(text))

    if has_grant and has_deny:
        return "partial", "granted_in_part_or_mixed"
    if has_grant:
        return "pass", "granted"
    if has_deny:
        return "fail", "denied"

    return None, None


def resolve_motion_outcome(
    docket_id,
    motion_type,
    entry_number,
    entry_date_filed,
    description,
    plain_text,
):
    """
    Determines whether a motion passed ("pass"), failed ("fail"), was partially
    granted ("partial"), or remains "pending_or_unknown".

    Why two steps are needed:
    PACER/CourtListener does not store a native boolean `passed` column on a motion.
    A party files a Motion in one docket entry, and the Judge rules on it in a
    Minute Entry / Order (either in the entry description, at the end of an order
    document, or in a subsequent docket entry on the same `docket_id`).
    """
    # 1. Check if the current docket entry description itself is an Order/Minute Entry
    desc_lower = (description or "").lower()
    is_order_or_minute = any(
        kw in desc_lower
        for kw in ("order", "minute entry", "opinion", "ruling", "granted", "denied")
    )
    if is_order_or_minute:
        label, detail = classify_ruling_text(description)
        if label:
            return label, detail, f"Docket description: {description.strip()}"

    # 2. If the document itself is a court order/opinion, check the last 2,500 chars
    if any(kw in desc_lower for kw in ("order", "opinion", "memorandum", "minute")):
        tail_text = (plain_text or "")[-2500:]
        label, detail = classify_ruling_text(tail_text)
        if label:
            return label, detail, f"Order text conclusion: {tail_text[-300:].strip()}"

    # 3. Search the same docket (`docket_id`) for a subsequent order/minute entry
    #    ruling on this motion (`granted` or `denied`).
    if CHECK_DOCKET_FOR_RULING and docket_id:
        time.sleep(POLITE_DELAY_SECONDS)
        # Build a targeted query on the same docket for rulings mentioning the motion
        # or its docket entry number
        entry_query = (
            f'("{motion_type}" OR "[{entry_number}]")'
            if entry_number
            else f'"{motion_type}"'
        )
        ruling_params = {
            "type": "rd",
            "q": (
                f"docket_id:{docket_id} AND {entry_query} "
                f'AND ("granted" OR "denied" OR "grants" OR "denies")'
            ),
        }
        if entry_date_filed:
            ruling_params["filed_after"] = entry_date_filed

        resp = request_with_retry(SEARCH_URL, headers=HEADERS, params=ruling_params)
        if resp is not None:
            ruling_hits = resp.json().get("results", [])
            for hit in ruling_hits:
                hit_desc = hit.get("description") or hit.get("short_description") or ""
                hit_snippet = re.sub(r"<[^>]+>", "", hit.get("snippet") or "")
                combined = f"{hit_desc} | {hit_snippet}".strip(" |")

                label, detail = classify_ruling_text(combined)
                if label:
                    hit_date = hit.get("entry_date_filed", "")
                    evidence = f"[{hit_date}] {combined[:350]}"
                    return label, detail, evidence

    return "pending_or_unknown", "unknown", ""


def save_json_records(records, filename):
    """Writes the list of motion records to a formatted UTF-8 JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ==============================================================================
# STEP 4: Load Previously Saved JSON Progress
# ==============================================================================

if os.path.exists(JSON_FILENAME):
    try:
        with open(JSON_FILENAME, "r", encoding="utf-8") as f:
            motion_records = json.load(f)
        print(f"Loaded {len(motion_records)} previously saved motions from {JSON_FILENAME}.")
    except Exception:
        motion_records = []
else:
    motion_records = []

seen_doc_ids = {
    row.get("doc_id") for row in motion_records if row.get("doc_id") is not None
}


# ==============================================================================
# STEP 5: Search 2016–Present Motions, Extract Full Text & Pass/Fail Labels
# ==============================================================================

for court_id, court_name in courts.items():
    for motion in motions:
        print(
            f"\nSearching {court_id} ({court_name}) for '{motion}' "
            f"filed {START_DATE} to {END_DATE}..."
        )

        # Query parameters for v4 Search API:
        # - type="r" returns dockets with nested `recap_documents`
        # - available_only="on" restricts results to filings with PDFs/text in RECAP
        # - filed_after / filed_before restrict filing dates to 2016-01-01..today
        # - q uses entry_date_filed:[2016-01-01 TO *] so nested documents also
        #   fall within 2016 to present
        params = {
            "q": f'"{motion}" AND entry_date_filed:[{START_DATE} TO {END_DATE}]',
            "type": "r",
            "court": court_id,
            "available_only": "on",
            "filed_after": START_DATE,
            "filed_before": END_DATE,
        }

        response = request_with_retry(SEARCH_URL, headers=HEADERS, params=params)
        if response is None:
            continue

        data = response.json()
        docket_count = data.get("count", 0)
        document_count = data.get("document_count", docket_count)

        print(
            f"✓ {court_id}: '{motion}' (2016–present) -> "
            f"~{docket_count} dockets (~{document_count} available filings)"
        )

        extracted_for_combo = 0
        for docket_hit in data.get("results", []):
            if extracted_for_combo >= MAX_DOCS_PER_COMBO:
                break

            case_name = docket_hit.get("caseName", "Unknown Case")
            docket_number = docket_hit.get("docketNumber", "")
            docket_id = docket_hit.get("docket_id")

            for doc in docket_hit.get("recap_documents", []):
                if extracted_for_combo >= MAX_DOCS_PER_COMBO:
                    break

                doc_id = doc.get("id")
                is_available = doc.get("is_available", False)
                filepath_local = doc.get("filepath_local") or ""
                entry_date = doc.get("entry_date_filed") or ""

                if not doc_id or doc_id in seen_doc_ids:
                    continue

                # Ensure the filing itself is from 2016 onward
                if entry_date and entry_date < START_DATE:
                    continue

                if not is_available and not filepath_local:
                    continue

                description = (
                    doc.get("description") or doc.get("short_description") or ""
                )
                entry_number = doc.get("entry_number")
                snippet = re.sub(r"<[^>]+>", "", doc.get("snippet") or "").strip()

                # 1. Fetch full extracted text (`plain_text`) via /recap-documents/{id}/
                doc_detail = fetch_recap_document_text(doc_id)
                plain_text = (doc_detail.get("plain_text") or "").strip()
                filepath_local = doc_detail.get("filepath_local") or filepath_local
                page_count = doc_detail.get("page_count", doc.get("page_count"))
                if not description:
                    description = doc_detail.get("description") or ""

                # 2. Determine whether the motion passed ("pass"), failed ("fail"),
                #    was partially granted ("partial"), or is "pending_or_unknown"
                outcome_label, outcome_detail, ruling_evidence = resolve_motion_outcome(
                    docket_id=docket_id,
                    motion_type=motion,
                    entry_number=entry_number,
                    entry_date_filed=entry_date,
                    description=description,
                    plain_text=plain_text,
                )

                pdf_url = (
                    f"{STORAGE_BASE_URL}{filepath_local.lstrip('/')}"
                    if filepath_local
                    else ""
                )

                # Optional: also download the binary PDF if enabled
                if DOWNLOAD_PDF_FILES and filepath_local:
                    pdf_filename = f"{court_id}_{doc_id}.pdf"
                    pdf_path = os.path.join(PDF_DIR, pdf_filename)
                    if not os.path.exists(pdf_path):
                        pdf_resp = request_with_retry(pdf_url, headers={}, stream=True)
                        if pdf_resp is not None:
                            with open(pdf_path, "wb") as pf:
                                for chunk in pdf_resp.iter_content(chunk_size=8192):
                                    if chunk:
                                        pf.write(chunk)

                record = {
                    "doc_id": doc_id,
                    "docket_id": docket_id,
                    "court_id": court_id,
                    "court_name": court_name,
                    "motion_type": motion,
                    "case_name": case_name,
                    "docket_number": docket_number,
                    "entry_number": entry_number,
                    "entry_date_filed": entry_date,
                    "page_count": page_count,
                    "description": description,
                    "outcome_label": outcome_label,     # "pass", "fail", "partial", or "pending_or_unknown"
                    "outcome_detail": outcome_detail,   # "granted", "denied", "granted_in_part", etc.
                    "ruling_evidence": ruling_evidence, # Docket entry / order text showing the ruling
                    "pdf_url": pdf_url,
                    "search_snippet": snippet,
                    "plain_text": plain_text,
                }

                motion_records.append(record)
                seen_doc_ids.add(doc_id)
                extracted_for_combo += 1

                # Save JSON immediately after each motion
                save_json_records(motion_records, JSON_FILENAME)

                print(
                    f"  -> Saved doc {doc_id} ({entry_date}) | "
                    f"outcome={outcome_label} ({outcome_detail}) | "
                    f"text={len(plain_text)} chars"
                )

        time.sleep(POLITE_DELAY_SECONDS)


# ==============================================================================
# STEP 6: Summary & JSON File Download
# ==============================================================================

print(f"\nTotal 2016–present motions saved to {JSON_FILENAME}: {len(motion_records)}")

if motion_records:
    # Print a quick breakdown of outcome labels
    outcome_counts = {}
    for rec in motion_records:
        lbl = rec.get("outcome_label", "pending_or_unknown")
        outcome_counts[lbl] = outcome_counts.get(lbl, 0) + 1
    print("Outcome label breakdown:", outcome_counts)

    if IN_COLAB:
        files.download(JSON_FILENAME)

print("Extraction complete!")