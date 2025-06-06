import re
from datetime import datetime
from dateutil import parser
from typing import List, Tuple

def normalize_month(text: str) -> str:
    # Correct common misspellings
    corrections = {
        'august': ['aug', 'august', 'augst', 'augest'],
        'january': ['jan', 'janurary'],
        'february': ['feb', 'febuary'],
        'september': ['sep', 'sept'],
        'october': ['oct', 'octomber'],
        'november': ['nov'],
        'december': ['dec'],
        'march': ['mar'],
        'april': ['apr'],
        'june': ['jun'],
        'july': ['jul'],
        'may': ['may'],
    }
    text = text.lower()
    for correct, variants in corrections.items():
        for v in variants:
            if v in text:
                return correct
    return text

def parse_date_safe(date_str: str) -> datetime:
    try:
        date_str = normalize_month(date_str)
        return parser.parse(date_str, fuzzy=True)
    except:
        return None

def extract_date_ranges(text: str) -> List[Tuple[datetime, datetime]]:
    text = text.lower()
    text = text.replace("till date", "present").replace("to date", "present").replace("to current", "present")

    patterns = [
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*\d{4})\s*[-–to]+\s*(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*[\s,]*(?:\d{4}|present))',
        r'(\d{4})\s*[-–to]+\s*(\d{4}|present)'
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    ranges = []
    for start_str, end_str in matches:
        start = parse_date_safe(start_str.strip())
        end = parse_date_safe(end_str.strip()) if 'present' not in end_str else datetime.today()
        if start and end and start < end:
            ranges.append((start, end))

    return ranges

def extract_experience_statements(text: str) -> float:
    """
    Parse lines like "Python- Exprience - 24 months" or "Exprience - Less than 1 year"
    """
    text = text.lower()
    total_months = 0

    # Match formats like "24 months", "2 years", "less than 1 year", etc.
    month_matches = re.findall(r'(\d+)\s*month', text)
    total_months += sum(int(m) for m in month_matches)

    year_matches = re.findall(r'(\d+)\s*year', text)
    total_months += sum(int(y) * 12 for y in year_matches)

    # Handle "less than 1 year" as 6 months conservatively
    if 'less than 1 year' in text:
        total_months += 6

    return total_months / 12

def estimate_experience_years(text: str) -> float:
    date_ranges = extract_date_ranges(text)
    months_from_dates = 0
    for start, end in date_ranges:
        months = (end.year - start.year) * 12 + (end.month - start.month)
        months_from_dates += max(months, 0)

    months_from_statements = extract_experience_statements(text)

    total_months = months_from_dates + months_from_statements * 12

    # === Fallback Logic ===
    if total_months == 0:
        fallback_clues = [
            r'currently working (as|in)',
            r'experience in\b',
            r'java developer\b',
            r'\bproject title\b',
            r'\brole:\s*java developer\b',
            r'worked on',
            r'\btools and technologies\b',
            r'\bcompany - '
        ]
        clue_hits = sum(bool(re.search(pattern, text.lower())) for pattern in fallback_clues)

        if clue_hits >= 3:
            # Conservative estimate: 2 years
            return 2.0

    return round(total_months / 12, 2)
