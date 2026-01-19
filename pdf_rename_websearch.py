# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PDF Renaming with WebSearch Verification
#
# ## 処理フロー
# 1. 全PDFからタイトルとメタデータを抽出
# 2. タイトルでWebSearchを実施し著者名を取得
# 3. WebSearch著者名とメタデータ著者名を比較
#    - 一致しない場合 → WebSearchの著者名を使用
#    - WebSearch著者名がNON_SURNAMESのみ → `_alert`をつけてPDFテキストサーチ
# 4. 出版年はメタデータ優先、なければ`_alert`でPDFサーチ
# 5. 最終的にNON_SURNAMESのみ or 出版年不明 → `_fail`フラグ

# %%
# Cell 1: Setup and Imports
import os
import re
import json
import time
import pypdf
import pdfplumber
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Directory containing PDFs
ARTICLES_DIR = os.getcwd()  # Change this if running from different location
print(f"Working directory: {ARTICLES_DIR}")
print(f"PDF count: {len([f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith('.pdf')])}")

# %%
# Cell 2: NON_SURNAMES List (Comprehensive)

NON_SURNAMES = {
    # Articles, prepositions, conjunctions
    'the', 'and', 'for', 'from', 'with', 'that', 'this', 'are', 'was', 'were',
    'have', 'has', 'been', 'being', 'can', 'could', 'would', 'should', 'will',
    'not', 'but', 'all', 'some', 'any', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'such', 'only', 'own', 'same', 'than', 'too', 'very',
    'of', 'in', 'on', 'to', 'by', 'at', 'or', 'an', 'as', 'if', 'so', 'no', 'up',
    'about', 'into', 'over', 'after', 'under', 'above', 'between', 'through',
    'during', 'before', 'against', 'among', 'throughout', 'despite', 'towards',
    'upon', 'concerning', 'regarding', 'unlike', 'beyond', 'within', 'without',
    
    # Pronouns and determiners
    'we', 'he', 'she', 'it', 'they', 'you', 'who', 'what', 'where', 'when',
    'why', 'how', 'which', 'whose', 'whom', 'their', 'them', 'these', 'those',
    
    # Academic/Journal terms
    'journal', 'review', 'quarterly', 'annual', 'american', 'economic', 'economics',
    'econometrica', 'econometric', 'econometrics', 'political', 'science', 'sciences',
    'statistical', 'statistics', 'sociological', 'sociology', 'association', 'society',
    'institute', 'university', 'department', 'school', 'college', 'press', 'publisher',
    'volume', 'vol', 'issue', 'number', 'page', 'pages', 'chapter', 'section', 'part',
    'series', 'working', 'paper', 'papers', 'discussion', 'research', 'study', 'studies',
    'analysis', 'abstract', 'introduction', 'conclusion', 'appendix', 'references',
    'bibliography', 'acknowledgments', 'contents', 'table', 'figure', 'notes',
    'editor', 'editors', 'editorial', 'author', 'authors', 'contributor', 'published',
    'publication', 'copyright', 'rights', 'reserved', 'reprint', 'reprinted',
    'forthcoming', 'manuscript', 'draft', 'revision', 'revised', 'version', 'online', 'print',
    
    # Common nouns that appeared as false positives
    'evidence', 'effect', 'effects', 'impact', 'policy', 'market', 'markets', 'labor',
    'trade', 'growth', 'development', 'innovation', 'technology', 'industry', 'firm',
    'firms', 'data', 'model', 'models', 'theory', 'method', 'methods', 'approach',
    'framework', 'system', 'systems', 'process', 'result', 'results', 'finding',
    'findings', 'outcome', 'outcomes', 'response', 'behavior', 'performance', 'quality',
    'value', 'price', 'cost', 'income', 'wage', 'wages', 'employment', 'unemployment',
    'education', 'health', 'welfare', 'social', 'public', 'private', 'national',
    'international', 'global', 'local', 'regional', 'urban', 'rural', 'human', 'capital',
    'investment', 'return', 'returns', 'risk', 'information', 'knowledge', 'learning',
    
    # Problematic extractions from previous runs
    'econ', 'educ', 'employed', 'equivalence', 'externalities', 'functions', 'goods',
    'implications', 'indicators', 'infants', 'inspections', 'interactions', 'introductory',
    'makers', 'modern', 'netherlands', 'novelty', 'obesity', 'openaccess', 'opportunities',
    'overconfidence', 'ownership', 'patterns', 'payments', 'peer', 'pharmaceuticals',
    'physics', 'pitfalls', 'platform', 'practices', 'procedures', 'producer', 'professors',
    'promise', 'promotions', 'psychology', 'puzzle', 'relationships', 'reply', 'road',
    'scientists', 'score', 'sectors', 'shock', 'shootings', 'soccer', 'solutions',
    'speed', 'spread', 'sports', 'spinoffs', 'station', 'statistician', 'structural',
    'street', 'students', 'symposium', 'talent', 'teacher', 'team', 'technical',
    'tenure', 'temperature', 'union', 'united', 'unobservables', 'closures', 'gasoline',
    'accuracy', 'semiconductor', 'texas', 'unemployed', 'benefit', 'dynamics', 'determinants',
    'office', 'past', 'patent', 'patents', 'reading', 'moment', 'networks', 'lecture',
    'contents', 'available', 'sciencedirect', 'springer', 'elsevier', 'wiley', 'oxford',
    
    # Truncated words
    'nference', 'nstruments', 'nteractionof', 'nthe', 'ntwo', 'nfluenceof',
    'omputers', 'ontrolcontrol', 'oodor', 'orhigh', 'orizontalscope',
    
    # Concatenated names (should be split)
    'jamesj', 'joshuad', 'johna', 'jonathanroth', 'kevinlewis', 'jasonkaufman',
    'ronj', 'patrickbajari', 'pedromira', 'orionpenner', 'peterc', 'petrae',
    'thomasj', 'imranrasul', 'isaiahandrews', 'morikazuushiogi',
    
    # Place names often in affiliations
    'boston', 'cambridge', 'chicago', 'stanford', 'berkeley', 'princeton', 'yale',
    'harvard', 'columbia', 'mit', 'nber', 'cepr', 'minneapolis', 'newyork',
    'california', 'massachusetts', 'pennsylvania', 'michigan', 'texas', 'florida',
    
    # More problematic patterns
    'advance', 'access', 'multivariate', 'behavioral', 'sinica', 'proc', 'natl',
    'acad', 'sci', 'usa', 'applied', 'quarterly', 'monthly', 'weekly', 'daily',
    'exhibit', 'exhibition', 'quint', 'portera', 'aroth', 'aplicada',
    'annals', 'friends', 'introductory', 'lecture', 'higher', 'educ',
    
    # First names that should not be used as surnames
    'david', 'john', 'michael', 'james', 'robert', 'william', 'richard', 'joseph',
    'thomas', 'charles', 'christopher', 'daniel', 'matthew', 'anthony', 'mark',
    'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth', 'kevin', 'brian',
    'george', 'timothy', 'ronald', 'edward', 'jason', 'jeffrey', 'ryan', 'jacob',
    'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott',
    'brandon', 'benjamin', 'samuel', 'raymond', 'gregory', 'frank', 'alexander',
    'patrick', 'jack', 'dennis', 'jerry', 'tyler', 'aaron', 'jose', 'adam',
    'nathan', 'henry', 'douglas', 'zachary', 'peter', 'kyle', 'noah', 'ethan',
    'jeremy', 'walter', 'christian', 'keith', 'roger', 'terry', 'austin', 'sean',
    'gerald', 'carl', 'harold', 'dylan', 'arthur', 'lawrence', 'jordan', 'jesse',
    'bryan', 'billy', 'bruce', 'gabriel', 'joe', 'logan', 'albert', 'willie',
    'alan', 'eugene', 'vincent', 'russell', 'elijah', 'randy', 'philip', 'harry',
    'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan',
    'jessica', 'sarah', 'karen', 'lisa', 'nancy', 'betty', 'margaret', 'sandra',
    'ashley', 'kimberly', 'emily', 'donna', 'michelle', 'dorothy', 'carol', 'amanda',
    'melissa', 'deborah', 'stephanie', 'rebecca', 'sharon', 'laura', 'cynthia',
    'kathleen', 'amy', 'angela', 'shirley', 'anna', 'brenda', 'pamela', 'emma',
    'nicole', 'helen', 'samantha', 'katherine', 'christine', 'debra', 'rachel',
    'carolyn', 'janet', 'catherine', 'maria', 'heather', 'diane', 'ruth', 'julie',
    'olivia', 'joyce', 'virginia', 'victoria', 'kelly', 'lauren', 'christina',
}

print(f"NON_SURNAMES list contains {len(NON_SURNAMES)} words")


# %%
# Cell 3: Helper Functions

def is_valid_surname(name: str) -> bool:
    """Check if a name looks like a valid surname."""
    if not name:
        return False
    
    name = name.strip()
    name = re.sub(r'[:\.,;!?]+$', '', name)
    
    if len(name) < 2 or len(name) > 20:
        return False
    
    if name.lower() in NON_SURNAMES:
        return False
    
    if any(c.isdigit() for c in name):
        return False
    
    letters = sum(1 for c in name if c.isalpha())
    if letters < len(name) * 0.8:
        return False
    
    # Reject words ending with common suffixes
    if re.search(r'(ing|tion|ment|ness|ity|ous|ive|able|ible|ence|ance|ology|ics)$', name.lower()):
        if len(name) > 8:
            return False
    
    return True


def normalize_case(name: str) -> Optional[str]:
    """Normalize name to proper case."""
    if not name:
        return None
    name = re.sub(r'[:\.,;!?]+$', '', name).strip()
    if not name:
        return None
    if name.isupper():
        return name.capitalize()
    if name[0].islower():
        name = name[0].upper() + name[1:]
    return name


def extract_title_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract the paper title from PDF first page."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None
            
            text = pdf.pages[0].extract_text() or ""
            if not text or len(text) < 50:
                return None
            
            # Check for unreadable text
            if len(re.findall(r'\(cid:\d+\)', text[:500])) > 5:
                return None
            
            lines = text.split('\n')
            
            for line in lines[:25]:
                line = line.strip()
                
                if len(line) < 15:
                    continue
                
                # Skip metadata lines
                skip_patterns = [
                    r'^(abstract|introduction|keywords|doi|http|www)',
                    r'^(volume|vol\.|issue|no\.|received|accepted|published)',
                    r'^(©|copyright|\d{4})',
                    r'^\d+$',
                    r'^[A-Z][A-Z\s\.]{2,30}$',
                    r'^[Bb][Yy]\s+',
                    r'(journal|review|quarterly|econometrica|american economic)',
                ]
                
                skip = False
                for pattern in skip_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        skip = True
                        break
                if skip:
                    continue
                
                if 20 <= len(line) <= 200:
                    title = re.sub(r'\s+', ' ', line).strip()
                    title = re.sub(r'[\d\*†‡§¶]+$', '', title).strip()
                    if len(title) >= 15:
                        return title
            
            return None
    except Exception as e:
        return None


def extract_metadata(pdf_path: str) -> Tuple[List[str], Optional[str]]:
    """Extract authors and year from PDF metadata."""
    authors = []
    year = None
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        meta = reader.metadata
        if not meta:
            return [], None
        
        # Authors
        author_field = meta.get('/Author', '') or ''
        if author_field:
            author_field = re.sub(r'[†‡*§¶∗]+', ' ', author_field)
            parts = re.split(r'[,;&]|\band\b', author_field, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if part and len(part) > 1:
                    words = part.split()
                    if words:
                        surname = words[-1] if len(words) > 1 else words[0]
                        surname = re.sub(r'[:\.,;!?]+$', '', surname)
                        if surname and len(surname) >= 2:
                            authors.append(surname)
        
        # Year from CreationDate
        for field in ['/CreationDate', '/ModDate']:
            val = meta.get(field, '') or ''
            match = re.search(r'D:(\d{4})', val)
            if match:
                y = int(match.group(1))
                if 1960 <= y <= 2026:
                    year = str(y)
                    break
        
        return authors, year
    except Exception:
        return [], None


def is_japanese_pdf(pdf_path: str) -> bool:
    """Check if PDF contains Japanese/Chinese characters."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return False
            text = pdf.pages[0].extract_text() or ""
            if len(text) < 50:
                return False
            # Check for Japanese/Chinese characters in first 500 chars
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text[:500]):
                return True
        return False
    except Exception:
        return False


print("Helper functions defined.")


# %%
# Cell 4: PDF Text Search Functions (Fallback for _alert cases)

def extract_authors_from_text(pdf_path: str) -> Tuple[List[str], Optional[str], bool, bool]:
    """
    Extract authors and year from PDF text.
    Returns: (authors, year, is_readable, is_japanese)
    """
    authors = []
    year = None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return [], None, False, False
            
            text = pdf.pages[0].extract_text() or ""
            if not text or len(text) < 50:
                return [], None, False, False
            
            # Check for unreadable text (cid format)
            cid_count = len(re.findall(r'\(cid:\d+\)', text[:2000]))
            if cid_count > 10:
                return [], None, False, False
            
            # Check for Japanese/Chinese characters
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text[:500]):
                return [], None, True, True  # is_readable=True, is_japanese=True
            
            lines = text.split('\n')
            
            # Find year
            for line in lines[:30]:
                # Skip data range lines
                if re.search(r'data|sample|period|survey', line, re.IGNORECASE):
                    continue
                # Skip year ranges
                if re.search(r'\b(19[6-9]\d|20[0-2]\d)[–—-](19[6-9]\d|20[0-2]\d)\b', line):
                    continue
                
                # Look for publication year patterns
                pub_match = re.search(r'(?:published|©|copyright|\(|received|accepted)[:\s]*(\d{4})', line, re.IGNORECASE)
                if pub_match:
                    y = int(pub_match.group(1))
                    if 1960 <= y <= 2026:
                        year = str(y)
                        break
                
                # Standalone year
                match = re.search(r'\b(19[6-9]\d|20[0-2]\d)\b', line)
                if match and not year:
                    year = match.group(1)
            
            # Find authors
            for line in lines[:25]:
                line = line.strip()
                if len(line) > 80 or len(line) < 5:
                    continue
                
                # "By Author" pattern
                by_match = re.match(r'^[Bb][Yy]\s+(.+)$', line)
                if by_match:
                    author_str = by_match.group(1)
                    parts = re.split(r'[,;&]|\band\b', author_str, flags=re.IGNORECASE)
                    for part in parts:
                        words = part.strip().split()
                        if words:
                            surname = normalize_case(words[-1])
                            if surname and is_valid_surname(surname):
                                authors.append(surname)
                    if authors:
                        break
                
                # "First M. Last" pattern
                match = re.match(r'^([A-Z][a-z]+)\s+([A-Z]\.?\s+)?([A-Z][a-z]+)[*†∗‡§¶]*$', line)
                if match and len(line) < 40:
                    surname = match.group(3)
                    if is_valid_surname(surname):
                        authors.append(surname)
                
                # ALL CAPS name pattern
                match = re.match(r'^([A-Z]{2,})\s+([A-Z]\.?\s*)?([A-Z]{2,})[*†∗‡§¶]*$', line)
                if match and len(line) < 35:
                    surname = normalize_case(match.group(3))
                    if surname and is_valid_surname(surname):
                        authors.append(surname)
            
            return authors, year, True, False  # is_readable=True, is_japanese=False
    
    except Exception as e:
        return [], None, False, False


print("PDF text search functions defined.")


# %%
# Cell 5: WebSearch Functions
# Primary: Semantic Scholar API
# Fallback: CrossRef API

import requests


def search_semantic_scholar(title: str, clean_title: str) -> Tuple[List[str], Optional[str], str]:
    """
    Search Semantic Scholar API.
    Returns: (authors, year, source)
    """
    try:
        time.sleep(1.0)  # Rate limit: 100 requests/5 min

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": clean_title,
            "fields": "title,authors,year",
            "limit": 5
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                for paper in data['data']:
                    paper_title = paper.get('title', '').lower()
                    if title[:30].lower() in paper_title or paper_title[:30] in title.lower():
                        authors = []
                        for author in paper.get('authors', []):
                            name = author.get('name', '')
                            if name:
                                words = name.split()
                                if words:
                                    surname = words[-1]
                                    authors.append(surname)
                        year = paper.get('year')
                        if authors:
                            return authors, str(year) if year else None, 'semantic_scholar'

        return [], None, ''

    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return [], None, ''


def search_crossref(title: str, clean_title: str) -> Tuple[List[str], Optional[str], str]:
    """
    Search CrossRef API (fallback).
    Returns: (authors, year, source)
    """
    try:
        time.sleep(0.5)  # Rate limit: be polite

        url = "https://api.crossref.org/works"
        params = {
            "query.title": clean_title,
            "rows": 5
        }
        headers = {
            "User-Agent": "PDFRenamer/1.0 (mailto:your-email@example.com)"  # CrossRef asks for this
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            for item in items:
                # Check title similarity
                item_title = item.get('title', [''])[0].lower() if item.get('title') else ''
                if not item_title:
                    continue
                if title[:30].lower() not in item_title and item_title[:30] not in title.lower():
                    continue

                authors = []
                for author in item.get('author', []):
                    family = author.get('family', '')
                    if family:
                        authors.append(family)

                year = None
                # Try different date fields
                for date_field in ['published-print', 'published-online', 'issued', 'created']:
                    if item.get(date_field) and item[date_field].get('date-parts'):
                        parts = item[date_field]['date-parts']
                        if parts and parts[0] and parts[0][0]:
                            year = str(parts[0][0])
                            break

                if authors:
                    return authors, year, 'crossref'

        return [], None, ''

    except Exception as e:
        print(f"CrossRef error: {e}")
        return [], None, ''


def websearch_authors(title: str) -> Tuple[List[str], Optional[str], str]:
    """
    Search for paper authors using the title.
    Primary: Semantic Scholar API
    Fallback: CrossRef API

    Returns: (authors_list, year, source)
    """
    if not title or len(title) < 10:
        return [], None, ''

    # Clean title for search
    clean_title = re.sub(r'[^\w\s\-\'\":]', ' ', title)
    clean_title = re.sub(r'\s+', ' ', clean_title).strip()[:100]

    # Try Semantic Scholar first
    authors, year, source = search_semantic_scholar(title, clean_title)
    if authors:
        return authors, year, source

    # Fallback to CrossRef
    authors, year, source = search_crossref(title, clean_title)
    if authors:
        return authors, year, source

    return [], None, ''


print("WebSearch functions defined. Using Semantic Scholar + CrossRef fallback.")

# %%
# Cell 6: Step 1 - Extract titles and metadata from all PDFs
# Japanese PDFs are detected here and marked for moving (skipped in WebSearch)

print(f"Starting extraction at {datetime.now()}")

pdf_files = [f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDFs")

pdf_data = []

for i, filename in enumerate(sorted(pdf_files)):
    if (i + 1) % 100 == 0:
        print(f"Extracting {i+1}/{len(pdf_files)}...")

    path = os.path.join(ARTICLES_DIR, filename)

    # Check for Japanese PDF first
    if is_japanese_pdf(path):
        pdf_data.append({
            'filename': filename,
            'title': None,
            'meta_authors': [],
            'meta_year': None,
            'websearch_authors': None,
            'websearch_year': None,
            'websearch_source': None,  # semantic_scholar or crossref
            'final_authors': None,
            'final_year': None,
            'status': 'japanese',
            'fail_reason': 'japanese'
        })
        continue

    title = extract_title_from_pdf(path)
    meta_authors, meta_year = extract_metadata(path)

    pdf_data.append({
        'filename': filename,
        'title': title,
        'meta_authors': meta_authors,
        'meta_year': meta_year,
        'websearch_authors': None,
        'websearch_year': None,
        'websearch_source': None,  # semantic_scholar or crossref
        'final_authors': None,
        'final_year': None,
        'status': 'pending'  # pending, success, alert, fail, japanese
    })

japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')
print(f"\nExtraction complete.")
print(f"Japanese PDFs (will be moved): {japanese_count}")
print(f"PDFs with title: {sum(1 for x in pdf_data if x['title'])}")
print(f"PDFs with metadata authors: {sum(1 for x in pdf_data if x['meta_authors'])}")
print(f"PDFs with metadata year: {sum(1 for x in pdf_data if x['meta_year'])}")

# %%
# Cell 7: Step 2 - WebSearch for all PDFs with titles
# WARNING: This will take a long time for many files!
# Adjust batch_size and add sleep for rate limiting
# Note: Japanese PDFs are skipped (already marked in Cell 6)

batch_size = 50  # Process in batches to save progress
save_interval = 50  # Save progress every N files

non_japanese = [x for x in pdf_data if x['status'] != 'japanese']
print(f"Starting WebSearch at {datetime.now()}")
print(f"Files to process: {sum(1 for x in non_japanese if x['title'])}")
print(f"Skipping {sum(1 for x in pdf_data if x['status'] == 'japanese')} Japanese PDFs")

for i, item in enumerate(pdf_data):
    # Skip Japanese PDFs
    if item['status'] == 'japanese':
        continue

    if item['websearch_authors'] is not None:  # Skip already processed
        continue

    if not item['title']:
        item['websearch_authors'] = []
        item['websearch_year'] = None
        item['websearch_source'] = None
        continue

    # Perform WebSearch (Semantic Scholar -> CrossRef fallback)
    ws_authors, ws_year, ws_source = websearch_authors(item['title'])
    item['websearch_authors'] = ws_authors
    item['websearch_year'] = ws_year
    item['websearch_source'] = ws_source if ws_source else None
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(pdf_data)}...")
    
    # Save progress periodically
    if (i + 1) % save_interval == 0:
        with open(os.path.join(ARTICLES_DIR, 'websearch_progress.json'), 'w') as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=2)
        print(f"Progress saved at {i+1}")

# Final save
with open(os.path.join(ARTICLES_DIR, 'websearch_progress.json'), 'w') as f:
    json.dump(pdf_data, f, ensure_ascii=False, indent=2)

print(f"\nWebSearch complete at {datetime.now()}")
print(f"Files with WebSearch results: {sum(1 for x in pdf_data if x['websearch_authors'])}")
print(f"  - From Semantic Scholar: {sum(1 for x in pdf_data if x.get('websearch_source') == 'semantic_scholar')}")
print(f"  - From CrossRef: {sum(1 for x in pdf_data if x.get('websearch_source') == 'crossref')}")

# %%
# Cell 8: Step 3 - Compare WebSearch results with metadata and determine final authors/year

print("Comparing WebSearch results with metadata...")

for item in pdf_data:
    filename = item['filename']
    ws_authors = item['websearch_authors'] or []
    ws_year = item['websearch_year']
    meta_authors = item['meta_authors'] or []
    meta_year = item['meta_year']
    
    # Filter valid surnames
    ws_valid = [normalize_case(a) for a in ws_authors if is_valid_surname(a)]
    meta_valid = [normalize_case(a) for a in meta_authors if is_valid_surname(a)]
    
    # Determine final authors
    # Priority: WebSearch > Metadata (per user request)
    if ws_valid:
        item['final_authors'] = ws_valid
        item['author_source'] = 'websearch'
    elif meta_valid:
        item['final_authors'] = meta_valid
        item['author_source'] = 'metadata'
    else:
        item['final_authors'] = []
        item['author_source'] = 'none'
        item['status'] = 'alert'  # Need PDF text search
    
    # Determine final year
    # Priority: Metadata year (per user request)
    if meta_year:
        item['final_year'] = meta_year
        item['year_source'] = 'metadata'
    elif ws_year:
        item['final_year'] = ws_year
        item['year_source'] = 'websearch'
    else:
        item['final_year'] = None
        item['year_source'] = 'none'
        if item['status'] != 'alert':
            item['status'] = 'alert'  # Need PDF text search for year
    
    # If we have valid authors and year, mark as success
    if item['final_authors'] and item['final_year']:
        item['status'] = 'success'

# Summary
success_count = sum(1 for x in pdf_data if x['status'] == 'success')
alert_count = sum(1 for x in pdf_data if x['status'] == 'alert')
pending_count = sum(1 for x in pdf_data if x['status'] == 'pending')

print(f"\nComparison complete:")
print(f"  Success (ready to rename): {success_count}")
print(f"  Alert (need PDF text search): {alert_count}")
print(f"  Pending: {pending_count}")

# %%
# Cell 9: Step 4 - PDF Text Search for _alert files

alert_files = [x for x in pdf_data if x['status'] == 'alert']
print(f"Processing {len(alert_files)} alert files with PDF text search...")

for i, item in enumerate(alert_files):
    if (i + 1) % 50 == 0:
        print(f"Processing {i+1}/{len(alert_files)}...")
    
    path = os.path.join(ARTICLES_DIR, item['filename'])
    # Note: Japanese PDFs are already filtered in Cell 6
    text_authors, text_year, is_readable, _ = extract_authors_from_text(path)

    # Filter valid surnames
    text_valid = [normalize_case(a) for a in text_authors if is_valid_surname(a)]
    
    # Update authors if we found valid ones
    if text_valid and not item['final_authors']:
        item['final_authors'] = text_valid
        item['author_source'] = 'pdf_text'
    
    # Update year if we found one
    if text_year and not item['final_year']:
        item['final_year'] = text_year
        item['year_source'] = 'pdf_text'
    
    # Check if we now have valid data
    if item['final_authors'] and item['final_year']:
        item['status'] = 'success'
    elif not is_readable:
        item['status'] = 'fail'  # Unreadable PDF
        item['fail_reason'] = 'unreadable'
    elif not item['final_authors']:
        item['status'] = 'fail'
        item['fail_reason'] = 'no_valid_authors'
    elif not item['final_year']:
        item['status'] = 'fail'
        item['fail_reason'] = 'no_valid_year'

# Final summary
success_count = sum(1 for x in pdf_data if x['status'] == 'success')
fail_count = sum(1 for x in pdf_data if x['status'] == 'fail')
alert_count = sum(1 for x in pdf_data if x['status'] == 'alert')
japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')

print(f"\nPDF text search complete:")
print(f"  Success: {success_count}")
print(f"  Fail: {fail_count}")
print(f"  Alert: {alert_count}")
print(f"  Japanese: {japanese_count}")


# %%
# Cell 10: Generate new filenames

def generate_filename(authors: List[str], year: str, existing: set) -> Optional[str]:
    """Generate filename from authors and year."""
    if not authors:
        return None
    
    authors = [normalize_case(a) for a in authors if a]
    authors = [a for a in authors if a and is_valid_surname(a)]
    
    if not authors:
        return None
    
    if len(authors) == 1:
        author_part = authors[0]
    elif len(authors) == 2:
        author_part = f"{authors[0]}_{authors[1]}"
    else:
        author_part = f"{authors[0]}_et_al"
    
    # Clean author part
    author_part = re.sub(r'[^\w\u00C0-\u024F]', '_', author_part)
    author_part = re.sub(r'_+', '_', author_part).strip('_')
    
    if author_part and author_part[0].islower():
        author_part = author_part[0].upper() + author_part[1:]
    
    year_part = year or "n.d."
    base = f"{author_part}_{year_part}"
    name = f"{base}.pdf"
    
    if name.lower() not in existing:
        return name
    
    for c in 'abcdefghijklmnopqrstuvwxyz':
        name = f"{base}_{c}.pdf"
        if name.lower() not in existing:
            return name
    
    return None


# Generate new names
existing = set()

for item in pdf_data:
    if item['status'] == 'success':
        new_name = generate_filename(item['final_authors'], item['final_year'], existing)
        if new_name:
            item['new_filename'] = new_name
            existing.add(new_name.lower())
        else:
            item['new_filename'] = None
            item['status'] = 'fail'
            item['fail_reason'] = 'filename_generation_failed'
    elif item['status'] == 'fail':
        # Add _fail suffix
        base = os.path.splitext(item['filename'])[0]
        base = re.sub(r'_(alert|fail)$', '', base)
        item['new_filename'] = f"{base}_fail.pdf"
    else:
        # Alert - shouldn't happen at this point
        base = os.path.splitext(item['filename'])[0]
        base = re.sub(r'_(alert|fail)$', '', base)
        item['new_filename'] = f"{base}_alert.pdf"

print(f"Filename generation complete.")
print(f"Files to rename: {sum(1 for x in pdf_data if x.get('new_filename'))}")

# %%
# Cell 11: Preview changes before renaming

print("=" * 60)
print("PREVIEW OF CHANGES")
print("=" * 60)

# Show some examples
success_items = [x for x in pdf_data if x['status'] == 'success' and x.get('new_filename')][:20]
fail_items = [x for x in pdf_data if x['status'] == 'fail'][:10]

print(f"\n--- Success examples (first 20) ---")
for item in success_items:
    print(f"{item['filename']}")
    print(f"  -> {item['new_filename']}")
    print(f"     Authors: {item['final_authors']} (source: {item.get('author_source')})")
    print(f"     Year: {item['final_year']} (source: {item.get('year_source')})")
    print()

print(f"\n--- Fail examples (first 10) ---")
for item in fail_items:
    print(f"{item['filename']}")
    print(f"  Reason: {item.get('fail_reason', 'unknown')}")
    print()

print(f"\n--- Summary ---")
print(f"Success: {sum(1 for x in pdf_data if x['status'] == 'success')}")
print(f"Fail: {sum(1 for x in pdf_data if x['status'] == 'fail')}")

# %%
# Cell 12: Execute renaming
# *** CAUTION: This will actually rename files! ***
# Make sure you have a backup!

EXECUTE_RENAME = False  # Set to True to actually rename files

if EXECUTE_RENAME:
    print("Starting file rename...")
    
    # Create japanese folder if needed
    japanese_dir = os.path.join(ARTICLES_DIR, 'japanese')
    if not os.path.exists(japanese_dir):
        os.makedirs(japanese_dir)
        print(f"Created directory: {japanese_dir}")
    
    renamed = []
    moved_japanese = []
    failed = []
    
    for item in pdf_data:
        old_name = item['filename']
        old_path = os.path.join(ARTICLES_DIR, old_name)
        
        # Handle Japanese papers - move to japanese folder
        if item['status'] == 'japanese':
            new_path = os.path.join(japanese_dir, old_name)
            try:
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_japanese.append({'old': old_name, 'new': f'japanese/{old_name}'})
                else:
                    failed.append({'old': old_name, 'new': f'japanese/{old_name}', 'reason': 'path conflict'})
            except Exception as e:
                failed.append({'old': old_name, 'new': f'japanese/{old_name}', 'reason': str(e)})
            continue
        
        # Handle regular renaming
        new_name = item.get('new_filename')
        
        if not new_name or old_name == new_name:
            continue
        
        new_path = os.path.join(ARTICLES_DIR, new_name)
        
        try:
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                renamed.append({'old': old_name, 'new': new_name})
            else:
                failed.append({'old': old_name, 'new': new_name, 'reason': 'path conflict'})
        except Exception as e:
            failed.append({'old': old_name, 'new': new_name, 'reason': str(e)})
    
    print(f"\nRename complete:")
    print(f"  Renamed: {len(renamed)}")
    print(f"  Moved to japanese/: {len(moved_japanese)}")
    print(f"  Failed: {len(failed)}")
    
    # Save results
    with open(os.path.join(ARTICLES_DIR, 'rename_results_final.json'), 'w') as f:
        json.dump({
            'renamed': renamed,
            'moved_japanese': moved_japanese,
            'failed': failed,
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
else:
    print("EXECUTE_RENAME is False. Set to True to actually rename files.")
    print("Make sure you have a backup before proceeding!")
    
    # Show preview of Japanese files
    japanese_files = [x for x in pdf_data if x['status'] == 'japanese']
    if japanese_files:
        print(f"\nJapanese files to be moved ({len(japanese_files)}):")
        for item in japanese_files[:10]:
            print(f"  {item['filename']} -> japanese/")

# %%
# Cell 13: Save final results

# Save all data
with open(os.path.join(ARTICLES_DIR, 'pdf_rename_data.json'), 'w') as f:
    json.dump(pdf_data, f, ensure_ascii=False, indent=2)

# Save summary
summary = {
    'total_pdfs': len(pdf_data),
    'success': sum(1 for x in pdf_data if x['status'] == 'success'),
    'fail': sum(1 for x in pdf_data if x['status'] == 'fail'),
    'alert': sum(1 for x in pdf_data if x['status'] == 'alert'),
    'japanese': sum(1 for x in pdf_data if x['status'] == 'japanese'),
    'author_sources': {
        'websearch': sum(1 for x in pdf_data if x.get('author_source') == 'websearch'),
        'metadata': sum(1 for x in pdf_data if x.get('author_source') == 'metadata'),
        'pdf_text': sum(1 for x in pdf_data if x.get('author_source') == 'pdf_text'),
        'none': sum(1 for x in pdf_data if x.get('author_source') == 'none'),
    },
    'websearch_sources': {
        'semantic_scholar': sum(1 for x in pdf_data if x.get('websearch_source') == 'semantic_scholar'),
        'crossref': sum(1 for x in pdf_data if x.get('websearch_source') == 'crossref'),
    },
    'year_sources': {
        'metadata': sum(1 for x in pdf_data if x.get('year_source') == 'metadata'),
        'websearch': sum(1 for x in pdf_data if x.get('year_source') == 'websearch'),
        'pdf_text': sum(1 for x in pdf_data if x.get('year_source') == 'pdf_text'),
        'none': sum(1 for x in pdf_data if x.get('year_source') == 'none'),
    },
    'fail_reasons': {},
    'timestamp': datetime.now().isoformat()
}

# Count fail reasons
for item in pdf_data:
    if item['status'] in ('fail', 'japanese'):
        reason = item.get('fail_reason', 'unknown')
        summary['fail_reasons'][reason] = summary['fail_reasons'].get(reason, 0) + 1

with open(os.path.join(ARTICLES_DIR, 'pdf_rename_summary.json'), 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Results saved:")
print(f"  - pdf_rename_data.json (full data)")
print(f"  - pdf_rename_summary.json (summary)")
print()
print(json.dumps(summary, indent=2))
