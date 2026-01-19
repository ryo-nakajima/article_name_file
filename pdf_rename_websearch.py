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
import shutil
import subprocess
import hashlib
import pypdf
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
import logging
from openai import OpenAI

warnings.filterwarnings('ignore')
# Suppress PDF parsing warnings
logging.getLogger('pypdf').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# === Incremental Processing Support ===
PROGRESS_FILE = 'pdf_processing_progress.json'
SAVE_INTERVAL = 50  # Save progress every N files


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file for identification."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def load_progress() -> Dict:
    """
    Load existing progress from file.
    Returns dict with structure:
    {
        'by_hash': {hash: {data}},  # Main lookup by file hash
        'hash_to_filename': {hash: filename},  # For display purposes
        'last_updated': timestamp
    }
    """
    progress_path = os.path.join(ARTICLES_DIR, PROGRESS_FILE)
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
                # Ensure required keys exist
                if 'by_hash' not in data:
                    data['by_hash'] = {}
                if 'hash_to_filename' not in data:
                    data['hash_to_filename'] = {}
                return data
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    return {'by_hash': {}, 'hash_to_filename': {}, 'last_updated': None}


def save_progress(progress: Dict):
    """Save progress to file."""
    progress_path = os.path.join(ARTICLES_DIR, PROGRESS_FILE)
    progress['last_updated'] = datetime.now().isoformat()
    try:
        with open(progress_path, 'w') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")


def cleanup_deleted_files(progress: Dict, current_hashes: set) -> int:
    """
    Remove entries for files that no longer exist.
    Returns number of entries removed.
    """
    deleted_count = 0
    hashes_to_remove = []

    for file_hash in progress['by_hash']:
        if file_hash not in current_hashes:
            hashes_to_remove.append(file_hash)

    for file_hash in hashes_to_remove:
        del progress['by_hash'][file_hash]
        if file_hash in progress['hash_to_filename']:
            del progress['hash_to_filename'][file_hash]
        deleted_count += 1

    return deleted_count


def should_process_extraction(progress: Dict, file_hash: str) -> bool:
    """Check if file needs extraction (new or previously failed)."""
    if file_hash not in progress['by_hash']:
        return True  # New file

    entry = progress['by_hash'][file_hash]
    status = entry.get('status', '')

    # Re-process if extraction failed or never completed
    if status in ('extraction_failed', 'pending', ''):
        return True

    # Skip if extraction was successful (has title or marked as japanese/no_title)
    return False


def should_process_websearch(progress: Dict, file_hash: str) -> bool:
    """Check if file needs WebSearch (extracted but not searched)."""
    if file_hash not in progress['by_hash']:
        return False  # Can't search without extraction

    entry = progress['by_hash'][file_hash]

    # Skip Japanese files
    if entry.get('status') == 'japanese':
        return False

    # Skip if no title extracted
    if not entry.get('title'):
        return False

    # Process if websearch not done or failed
    websearch_status = entry.get('websearch_status', '')
    if websearch_status in ('done', 'not_found'):
        return False  # Already searched

    return True

# Check and install system dependencies (poppler, tesseract)
def check_and_install_dependencies():
    """Check if poppler and tesseract are installed, install via Homebrew if missing."""
    dependencies = {
        'pdftoppm': 'poppler',    # pdftoppm is part of poppler
        'tesseract': 'tesseract'
    }
    
    for cmd, package in dependencies.items():
        if shutil.which(cmd) is None:
            print(f"Installing {package} (required for OCR)...")
            try:
                result = subprocess.run(
                    ['brew', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print(f"  {package} installed successfully")
                else:
                    print(f"  Failed to install {package}: {result.stderr}")
            except FileNotFoundError:
                print(f"  Homebrew not found. Please install {package} manually: brew install {package}")
            except subprocess.TimeoutExpired:
                print(f"  Installation timed out for {package}")
        else:
            print(f"{cmd}: OK")

check_and_install_dependencies()

# OpenAI API setup
openai_client = OpenAI()  # Uses OPENAI_API_KEY from environment
GPT_MODEL = "gpt-4o-mini"  # Cost-effective model for this task

# Directory containing PDFs
ARTICLES_DIR = os.getcwd()  # Change this if running from different location
print(f"Working directory: {ARTICLES_DIR}")
print(f"PDF count: {len([f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith('.pdf')])}")
print(f"OpenAI API: Ready (model: {GPT_MODEL})")

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

    # International first names (European, Russian, etc.)
    'dmitry', 'guido', 'birte', 'pedro', 'pablo', 'miguel', 'jose', 'carlos', 'juan',
    'marco', 'luigi', 'giuseppe', 'francesco', 'antonio', 'hans', 'fritz', 'klaus',
    'wolfgang', 'heinrich', 'helmut', 'stefan', 'andreas', 'matthias', 'christoph',
    'pierre', 'jean', 'francois', 'jacques', 'philippe', 'michel', 'olivier', 'yves',
    'ivan', 'sergei', 'sergey', 'vladimir', 'nikolai', 'alexei', 'andrei', 'yuri',
    'boris', 'viktor', 'oleg', 'maxim', 'igor', 'pavel', 'konstantin', 'artem',
    'alyssa', 'timothy', 'jonathan', 'richard', 'rafael', 'xavier', 'fernando',

    # More affiliation/institution words
    'administration', 'administrative', 'administrator', 'administrators', 'faculty',
    'professor', 'assistant', 'associate', 'graduate', 'undergraduate', 'postgraduate',
    'doctoral', 'postdoc', 'postdoctoral', 'fellow', 'fellows', 'fellowship', 'center',
    'centre', 'laboratory', 'lab', 'group', 'division', 'unit', 'program', 'programme',
    'speyer', 'germany', 'france', 'italy', 'spain', 'switzerland', 'austria', 'belgium',
    'neurology', 'medicine', 'medical', 'clinical', 'hospital', 'clinic',

    # Title words that should not be surnames
    'uncertainty', 'herding', 'bias', 'citation', 'citationbias', 'anchoring',
    'influence', 'expert', 'opinion', 'correlation', 'causation', 'inference',
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

    # Reject garbled/truncated names: unusual starting consonant clusters
    # Be careful not to reject valid African/Asian names like "Ng", "Ndegwa"
    lower_name = name.lower()
    if len(lower_name) >= 3:
        # Check for clearly invalid 3+ consonant clusters at start
        invalid_starts = ['ndf', 'xzx', 'zxz', 'qxq', 'vbv', 'bvb', 'kmt', 'pmt', 'bnt', 'dnt', 'fnt', 'gnt', 'hnt', 'lnt', 'mnt', 'pnt', 'rnt', 'tnt', 'wnt', 'znt']
        for inv in invalid_starts:
            if lower_name.startswith(inv):
                return False
        # Also reject if starts with more than 3 consonants in a row
        if re.match(r'^[bcdfghjklmnpqrstvwxz]{4,}', lower_name):
            return False

    # Check minimum vowel ratio for longer names (names need vowels!)
    if len(lower_name) >= 5:
        vowels = sum(1 for c in lower_name if c in 'aeiou')
        if vowels < len(lower_name) * 0.15:  # Less than 15% vowels is suspicious
            return False

    # Reject if starts with lowercase (after stripping)
    if name and name[0].islower():
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


def find_title_candidates(page, max_candidates: int = 5) -> List[Dict]:
    """
    Find title candidates using multiple indicators with flexible scoring.
    Returns top N candidates sorted by score for GPT validation.

    Indicators:
    - Font size (larger = higher score)
    - Bold font (bonus if detected)
    - ALL CAPS (bonus if detected)
    - Position (upper 5-35% of page, NOT at very top which is often header/journal name)
    - Appropriate length
    - Multi-line (suggests real title, not header)

    Returns list of candidate dicts with 'text' and 'score' keys.
    """
    try:
        words = page.extract_words(extra_attrs=['fontname', 'size'])
        if not words:
            return None

        page_height = page.height

        # Group words by line (similar y position)
        lines_with_info = []
        current_line = []
        current_top = None
        tolerance = 5  # pixels tolerance for same line

        for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
            if current_top is None or abs(word['top'] - current_top) <= tolerance:
                current_line.append(word)
                current_top = word['top'] if current_top is None else current_top
            else:
                if current_line:
                    lines_with_info.append(current_line)
                current_line = [word]
                current_top = word['top']

        if current_line:
            lines_with_info.append(current_line)

        # Calculate info for each line
        line_data = []
        for line_words in lines_with_info:
            text = ' '.join(w['text'] for w in line_words)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue

            avg_font_size = sum(w.get('size', 10) for w in line_words) / len(line_words)
            avg_top = sum(w['top'] for w in line_words) / len(line_words)
            avg_bottom = sum(w['bottom'] for w in line_words) / len(line_words)

            # Detect bold from fontname
            fontnames = [w.get('fontname', '') for w in line_words]
            is_bold = any('bold' in fn.lower() or 'heavy' in fn.lower() or
                         '-b' in fn.lower() or 'bd' in fn.lower()
                         for fn in fontnames if fn)

            # Detect ALL CAPS (at least 80% uppercase letters)
            alpha_chars = [c for c in text if c.isalpha()]
            is_all_caps = len(alpha_chars) > 5 and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) >= 0.8

            line_data.append({
                'text': text,
                'font_size': avg_font_size,
                'top': avg_top,
                'bottom': avg_bottom,
                'is_bold': is_bold,
                'is_all_caps': is_all_caps,
            })

        if not line_data:
            return None

        # Merge consecutive lines with similar characteristics (multi-line titles)
        # Merge if: similar font size, same bold status, reasonable gap, upper part of page
        font_tolerance = 2
        line_gap_tolerance = 30  # max gap between lines in pixels

        merged_candidates = []
        i = 0
        while i < len(line_data):
            current = line_data[i]
            merged_text = current['text']
            merged_font = current['font_size']
            merged_top = current['top']
            end_bottom = current['bottom']
            merged_bold = current['is_bold']
            merged_caps = current['is_all_caps']

            # Try to merge with following lines
            j = i + 1
            while j < len(line_data):
                next_line = line_data[j]
                gap = next_line['top'] - end_bottom

                # Check if should merge: similar font size, reasonable gap, both in upper part
                # Also prefer merging lines with same style (bold/caps)
                style_match = (next_line['is_bold'] == merged_bold or
                              next_line['is_all_caps'] == merged_caps)

                if (abs(next_line['font_size'] - merged_font) <= font_tolerance and
                    gap <= line_gap_tolerance and
                    next_line['top'] / page_height < 0.4 and
                    style_match):

                    merged_text += ' ' + next_line['text']
                    end_bottom = next_line['bottom']
                    # Keep bold/caps status if any merged line has it
                    merged_bold = merged_bold or next_line['is_bold']
                    merged_caps = merged_caps or next_line['is_all_caps']
                    j += 1
                else:
                    break

            merged_candidates.append({
                'text': merged_text,
                'font_size': merged_font,
                'top': merged_top,
                'line_count': j - i,
                'is_bold': merged_bold,
                'is_all_caps': merged_caps,
            })
            i = j

        # Score each candidate
        candidates = []
        for item in merged_candidates:
            text = item['text']

            if len(text) < 15 or len(text) > 300:
                continue

            # Skip metadata patterns (but not ALL CAPS titles)
            skip_patterns = [
                r'^(abstract|introduction|keywords|doi|http|www)',
                r'^(volume|vol\.|issue|no\.|received|accepted|published)',
                r'^(©|copyright|\d{4})',
                r'^\d+$',
                r'^[Bb][Yy]\s+',
            ]
            # Journal patterns - skip only if short (likely headers, not titles)
            if len(text) < 50:
                skip_patterns.append(r'^(journal|review|quarterly|econometrica|american economic)')

            skip = False
            for pattern in skip_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            position_ratio = item['top'] / page_height

            # === FLEXIBLE SCORING: Each indicator adds independently ===
            total_score = 0

            # 1. Font size score (max 25 points)
            # Larger fonts are more likely to be titles
            font_score = min((item['font_size'] - 8) / 16, 1.0) * 25
            total_score += max(0, font_score)

            # 2. Position score (max 30 points)
            # Optimal zone: 5-35% from top (not at very top = header area)
            if position_ratio < 0.05:
                # Very top - likely journal header, penalize
                position_score = 5
            elif position_ratio < 0.35:
                # Sweet spot for titles
                # Best score at around 10-20% from top
                if position_ratio < 0.20:
                    position_score = 30
                else:
                    position_score = 30 - (position_ratio - 0.20) * 50
            else:
                # Below 35% - decreasing score
                position_score = max(0, 20 - (position_ratio - 0.35) * 60)
            total_score += position_score

            # 3. Bold bonus (15 points if bold)
            if item['is_bold']:
                total_score += 15

            # 4. ALL CAPS bonus (10 points if all caps AND reasonable length)
            # Short ALL CAPS might be section headers, but longer ones are likely titles
            if item['is_all_caps'] and len(text) >= 30:
                total_score += 10

            # 5. Length score (max 15 points)
            if 40 <= len(text) <= 150:
                length_score = 15
            elif 25 <= len(text) < 40 or 150 < len(text) <= 200:
                length_score = 10
            else:
                length_score = 5
            total_score += length_score

            # 6. Multi-line bonus (5 points)
            # Multi-line text in title zone is very likely a title
            if item['line_count'] > 1:
                total_score += 5

            candidates.append({
                'score': total_score,
                'text': text,
                'font_size': item['font_size'],
                'position': position_ratio,
                'is_bold': item['is_bold'],
                'is_all_caps': item['is_all_caps'],
                'lines': item['line_count'],
            })

        if not candidates:
            return []

        # Sort by score descending
        candidates.sort(key=lambda x: -x['score'])

        # Clean up text and return top candidates
        result = []
        for c in candidates[:max_candidates]:
            title = re.sub(r'[\d\*†‡§¶]+$', '', c['text']).strip()
            if len(title) >= 15:
                result.append({
                    'text': title,
                    'score': c['score'],
                    'is_bold': c.get('is_bold', False),
                    'is_all_caps': c.get('is_all_caps', False),
                })
        return result

    except Exception:
        return []


def find_title_by_font_and_position(page) -> Optional[str]:
    """
    Legacy wrapper - returns single best candidate.
    Use find_title_candidates() for multiple candidates.
    """
    candidates = find_title_candidates(page, max_candidates=1)
    return candidates[0]['text'] if candidates else None


def call_gpt_with_retry(messages: list, max_tokens: int = 10,
                        max_retries: int = 3, base_delay: float = 10.0):
    """
    Call GPT API with retry logic for rate limit errors.

    Args:
        messages: Chat messages to send
        max_tokens: Maximum tokens in response
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries (doubles each retry)

    Returns:
        Response object or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            error_str = str(e)
            # Check for rate limit error (429)
            if '429' in error_str or 'rate_limit' in error_str.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 10s, 20s, 40s
                    print(f"  GPT rate limit hit, waiting {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  GPT rate limit: giving up after {max_retries} retries")
                    return None
            else:
                # Other errors, don't retry
                print(f"  GPT error: {e}")
                return None
    return None


def validate_title_with_gpt(candidate: str, pdf_text: str, max_text_chars: int = 2000) -> bool:
    """
    Use GPT to validate if a candidate string is likely the paper title.

    Args:
        candidate: The candidate title string
        pdf_text: Context from the PDF (first page text)
        max_text_chars: Max chars of PDF text to send

    Returns:
        True if GPT confirms this is likely the title, False otherwise
    """
    if not candidate or len(candidate) < 10:
        return False

    truncated_text = pdf_text[:max_text_chars] if pdf_text else ""

    messages = [
        {
            "role": "system",
            "content": """You are an expert at identifying academic paper titles.
Given a candidate title and the beginning of a PDF, determine if the candidate is the actual paper title.

Rules:
- Paper titles are usually descriptive of the research content
- They should NOT be journal names (e.g., "American Economic Review")
- They should NOT be section headers (e.g., "Abstract", "Introduction")
- They should NOT be author names or affiliations
- They should NOT be metadata (volume, issue, DOI, dates)

Respond with ONLY "YES" or "NO"."""
        },
        {
            "role": "user",
            "content": f"Candidate title: \"{candidate}\"\n\nPDF text context:\n{truncated_text}\n\nIs this the paper title?"
        }
    ]

    response = call_gpt_with_retry(messages)

    if response:
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
    else:
        # On error, accept the candidate (fallback)
        return True


def find_title_with_gpt_validation(page, pdf_text: str, use_gpt: bool = True) -> Optional[str]:
    """
    Find title by extracting candidates and validating with GPT.

    Args:
        page: pdfplumber page object
        pdf_text: Full text from the page for GPT context
        use_gpt: Whether to use GPT validation (set False to skip)

    Returns:
        Validated title or None
    """
    candidates = find_title_candidates(page, max_candidates=5)

    if not candidates:
        return None

    if not use_gpt:
        # Return best candidate without GPT validation
        return candidates[0]['text']

    # Try each candidate with GPT validation
    for candidate in candidates:
        title = candidate['text']
        if validate_title_with_gpt(title, pdf_text):
            return title

    # If all candidates rejected, return best one anyway (GPT might be wrong)
    # But add a lower confidence flag could be added here
    return candidates[0]['text']


def extract_title_from_pdf(pdf_path: str, use_ocr_fallback: bool = True, use_gpt: bool = True) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Extract the paper title from PDF first page.
    Uses font size and position information for better detection.
    Optionally validates with GPT.

    Returns: (title, method, error)
        - title: extracted title or None
        - method: 'text', 'text_gpt', 'ocr', 'ocr_gpt', or 'none'
        - error: error type if failed, None if success
    """
    error_reason = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                error_reason = 'no_pages'
            else:
                page = pdf.pages[0]
                text = page.extract_text() or ""

                # Check if text extraction failed
                if not text or len(text) < 50:
                    error_reason = 'no_text'
                # Check for unreadable text (cid format)
                elif len(re.findall(r'\(cid:\d+\)', text[:500])) > 5:
                    error_reason = 'cid_format'
                else:
                    # Try font/position-based detection with GPT validation
                    title = find_title_with_gpt_validation(page, text, use_gpt=use_gpt)
                    if title:
                        method = 'text_gpt' if use_gpt else 'text'
                        return title, method, None

                    # Fallback to simple line-based detection (without GPT for speed)
                    lines = text.split('\n')
                    for line in lines[:25]:
                        line = line.strip()
                        if len(line) < 15:
                            continue

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
                                return title, 'text', None

                    error_reason = 'no_title_found'

    except Exception:
        error_reason = 'pdf_read_error'

    # Try OCR if needed and enabled
    if error_reason and use_ocr_fallback:
        ocr_title = extract_title_with_ocr(pdf_path, use_gpt=use_gpt)
        if ocr_title:
            method = 'ocr_gpt' if use_gpt else 'ocr'
            return ocr_title, method, None
        else:
            return None, 'none', 'ocr_failed'

    return None, 'none', error_reason


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


def find_title_candidates_ocr(pdf_path: str, max_pages: int = 2, max_candidates: int = 5) -> Tuple[List[Dict], str]:
    """
    Extract title candidates from PDF using OCR.
    Returns candidates with scores for GPT validation.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to OCR
        max_candidates: Maximum candidates to return

    Returns:
        (list of candidate dicts, ocr_text for context)
    """
    try:
        # Convert first pages to images (200 DPI for good OCR quality)
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=200)
        if not images:
            return [], ""

        all_candidates = []
        ocr_text = ""

        # Try each page
        for page_num, image in enumerate(images, 1):
            # Use image_to_data to get bounding box information
            data = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DICT)

            if not data or 'text' not in data:
                continue

            # Also get plain text for GPT context
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_num == 1:
                ocr_text = page_text

            page_height = image.height

            # Group words by line (using block_num and line_num)
            lines_dict = {}
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue

                block = data['block_num'][i]
                line = data['line_num'][i]
                key = (block, line)

                if key not in lines_dict:
                    lines_dict[key] = {
                        'words': [],
                        'tops': [],
                        'heights': [],
                    }

                lines_dict[key]['words'].append(text)
                lines_dict[key]['tops'].append(data['top'][i])
                lines_dict[key]['heights'].append(data['height'][i])

            if not lines_dict:
                continue

            # Calculate line info
            line_data = []
            for key in sorted(lines_dict.keys()):
                info = lines_dict[key]
                text = ' '.join(info['words'])
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    continue

                avg_top = sum(info['tops']) / len(info['tops'])
                avg_height = sum(info['heights']) / len(info['heights'])

                # Detect ALL CAPS (at least 80% uppercase letters)
                alpha_chars = [c for c in text if c.isalpha()]
                is_all_caps = len(alpha_chars) > 5 and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) >= 0.8

                line_data.append({
                    'text': text,
                    'top': avg_top,
                    'height': avg_height,
                    'block': key[0],
                    'is_all_caps': is_all_caps,
                })

            if not line_data:
                continue

            # Merge consecutive lines (multi-line titles)
            height_tolerance_ratio = 0.3
            line_gap_tolerance = 50

            merged_candidates = []
            i = 0
            while i < len(line_data):
                current = line_data[i]
                merged_text = current['text']
                merged_top = current['top']
                merged_height = current['height']
                current_block = current['block']
                merged_caps = current['is_all_caps']
                end_bottom = current['top'] + current['height']

                j = i + 1
                while j < len(line_data):
                    next_line = line_data[j]
                    gap = next_line['top'] - end_bottom
                    height_diff = abs(next_line['height'] - merged_height) / max(merged_height, 1)
                    caps_match = (next_line['is_all_caps'] == merged_caps)

                    if (next_line['block'] == current_block and
                        height_diff <= height_tolerance_ratio and
                        gap <= line_gap_tolerance and
                        next_line['top'] / page_height < 0.4 and
                        caps_match):

                        merged_text += ' ' + next_line['text']
                        end_bottom = next_line['top'] + next_line['height']
                        merged_caps = merged_caps or next_line['is_all_caps']
                        j += 1
                    else:
                        break

                merged_candidates.append({
                    'text': merged_text,
                    'top': merged_top,
                    'height': merged_height,
                    'line_count': j - i,
                    'is_all_caps': merged_caps,
                })
                i = j

            # Score each candidate
            for item in merged_candidates:
                text = item['text']

                if len(text) < 15 or len(text) > 300:
                    continue

                skip_patterns = [
                    r'^(abstract|introduction|keywords|doi|http|www)',
                    r'^(volume|vol\.|issue|no\.|received|accepted|published)',
                    r'^(©|copyright|\d{4})',
                    r'^\d+$',
                    r'^[Bb][Yy]\s+',
                ]
                if len(text) < 50:
                    skip_patterns.append(r'^(journal|review|quarterly|econometrica|american economic)')

                skip = False
                for pattern in skip_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        skip = True
                        break
                if skip:
                    continue

                position_ratio = item['top'] / page_height

                # Scoring
                total_score = 0
                height_score = min((item['height'] - 15) / 40, 1.0) * 25
                total_score += max(0, height_score)

                if position_ratio < 0.05:
                    position_score = 5
                elif position_ratio < 0.35:
                    position_score = 30 if position_ratio < 0.20 else 30 - (position_ratio - 0.20) * 50
                else:
                    position_score = max(0, 20 - (position_ratio - 0.35) * 60)
                total_score += position_score

                if item['is_all_caps'] and len(text) >= 30:
                    total_score += 15

                if 40 <= len(text) <= 150:
                    total_score += 15
                elif 25 <= len(text) < 40 or 150 < len(text) <= 200:
                    total_score += 10
                else:
                    total_score += 5

                if item['line_count'] > 1:
                    total_score += 5

                # Clean up title text
                clean_title = re.sub(r'[\d\*†‡§¶]+$', '', text).strip()
                if len(clean_title) >= 15:
                    all_candidates.append({
                        'text': clean_title,
                        'score': total_score,
                        'is_all_caps': item['is_all_caps'],
                        'page': page_num,
                    })

            # If we have candidates from page 1, that's usually enough
            if all_candidates and page_num == 1:
                break

        # Sort all candidates by score
        all_candidates.sort(key=lambda x: -x['score'])
        return all_candidates[:max_candidates], ocr_text

    except Exception as e:
        return [], ""


def extract_title_with_ocr(pdf_path: str, max_pages: int = 2, use_gpt: bool = True) -> Optional[str]:
    """
    Extract title from PDF using OCR with optional GPT validation.
    Legacy wrapper for find_title_candidates_ocr + GPT validation.
    """
    candidates, ocr_text = find_title_candidates_ocr(pdf_path, max_pages=max_pages)

    if not candidates:
        return None

    if not use_gpt:
        return candidates[0]['text']

    # Try each candidate with GPT validation
    for candidate in candidates:
        if validate_title_with_gpt(candidate['text'], ocr_text):
            return candidate['text']

    # If all rejected, return best candidate
    return candidates[0]['text']


print("Helper functions defined.")


# %%
# Cell 3b: GPT-based Functions (Title Extraction and Author Validation)

def extract_title_with_gpt(text: str, max_chars: int = 3000) -> Optional[str]:
    """
    Use GPT to extract the paper title from PDF text.
    This is used as a fallback or verification for rule-based extraction.

    Args:
        text: The first portion of PDF text (usually first page)
        max_chars: Maximum characters to send to GPT (cost control)

    Returns:
        Extracted title or None if failed
    """
    if not text or len(text) < 50:
        return None

    # Truncate text to control costs
    truncated_text = text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": """You are an expert at identifying academic paper titles.
Given the beginning text of a PDF, extract ONLY the paper title.
Rules:
- Return ONLY the title, nothing else
- The title is usually in larger font, near the top (but not the very top which is often journal name)
- Do NOT include author names, journal names, volume/issue numbers, or dates
- If you cannot find a clear title, return "NONE"
- Return the title exactly as written (preserve capitalization)"""
        },
        {
            "role": "user",
            "content": f"Extract the paper title from this text:\n\n{truncated_text}"
        }
    ]

    response = call_gpt_with_retry(messages, max_tokens=200)

    if not response:
        return None

    title = response.choices[0].message.content.strip()

    if title == "NONE" or len(title) < 10:
        return None

    # Clean up the title
    title = re.sub(r'^["\'"]|["\'"]$', '', title)  # Remove quotes
    title = re.sub(r'[\d\*†‡§¶]+$', '', title).strip()

    return title if len(title) >= 15 else None


def validate_surnames_with_gpt(names: List[str]) -> List[str]:
    """
    Use GPT to validate and filter a list of potential surnames.
    Returns only names that are likely valid academic author surnames.

    Args:
        names: List of potential surname strings

    Returns:
        Filtered list of valid surnames
    """
    if not names:
        return []

    # Pre-filter obvious non-surnames
    filtered = [n for n in names if n and len(n) >= 2 and len(n) <= 25]
    if not filtered:
        return []

    names_str = ", ".join(filtered)

    messages = [
        {
            "role": "system",
            "content": """You are an expert at identifying valid academic author surnames.
Given a list of potential surnames, return ONLY the ones that are plausible human surnames.

Rules:
- Remove common words (the, and, for, with, etc.)
- Remove journal/publication terms (journal, review, economics, etc.)
- Remove institutional names (university, institute, etc.)
- Remove first names used alone (David, John, Mary, etc.)
- Keep valid surnames from any cultural background (Western, Asian, etc.)
- Format: Return valid surnames as comma-separated list
- If none are valid, return "NONE"

Examples:
Input: "Smith, Economics, Journal, Chen, Labor"
Output: "Smith, Chen"

Input: "The, And, Review, Markets"
Output: "NONE" """
        },
        {
            "role": "user",
            "content": f"Validate these potential surnames: {names_str}"
        }
    ]

    response = call_gpt_with_retry(messages, max_tokens=200)

    if not response:
        return names  # Return original list on error

    result = response.choices[0].message.content.strip()

    if result == "NONE":
        return []

    # Parse the comma-separated result
    validated = [n.strip() for n in result.split(",") if n.strip()]

    # Normalize case
    validated = [normalize_case(n) for n in validated if n]
    validated = [n for n in validated if n]

    return validated


def get_title_and_authors_with_gpt(text: str, max_chars: int = 4000) -> Tuple[Optional[str], List[str], Optional[str]]:
    """
    Use GPT to extract both title and authors from PDF text in a single call.
    More efficient than separate calls when both are needed.
    Includes retry logic for rate limit errors.

    Args:
        text: The first portion of PDF text

    Returns:
        (title, authors_list, year)
    """
    if not text or len(text) < 50:
        return None, [], None

    truncated_text = text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": """You are an expert at extracting metadata from academic papers.
Given the beginning text of a PDF, extract:
1. Paper title
2. Author surnames (last names only)
3. Publication year (if visible)

Return as JSON with this exact format:
{"title": "Paper Title Here", "authors": ["Smith", "Chen", "Lee"], "year": "2023"}

Rules for TITLE:
- The title is usually the LARGEST text near the top of the paper
- NOT the journal name (which appears at very top or in header)
- NOT copyright text like "Your use of the JSTOR archive indicates..."
- NOT sentences from the paper body (titles don't have words like "we estimate", "taken together", "this paper")
- If text has no spaces (concatenated like "Thisisthetitle"), try to find a properly spaced title elsewhere
- Return null if you cannot find a clear, proper title

Rules for AUTHORS:
- Extract ONLY surnames/last names (e.g., "Smith" not "John Smith")
- Authors often appear right below the title in SMALL CAPS or regular text
- Look for patterns like "By FIRSTNAME LASTNAME" or "FIRSTNAME LASTNAME*"
- Do NOT include journal names, institutions, or affiliations as authors
- Return empty array [] if no clear author names found

Rules for YEAR:
- The publication year, usually 4 digits (1990-2026)
- Often appears near copyright notice or in header/footer"""
        },
        {
            "role": "user",
            "content": f"Extract metadata from this paper:\n\n{truncated_text}"
        }
    ]

    # Retry logic for rate limits
    max_retries = 3
    base_delay = 10.0

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            title = result.get("title")
            if title and len(title) < 15:
                title = None

            authors = result.get("authors", [])
            if isinstance(authors, list):
                authors = [normalize_case(a) for a in authors if a and isinstance(a, str)]
                authors = [a for a in authors if a and is_valid_surname(a)]
            else:
                authors = []

            year = result.get("year")
            if year:
                year = str(year)
                if not re.match(r'^(19[6-9]\d|20[0-2]\d)$', year):
                    year = None

            return title, authors, year

        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate_limit' in error_str.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  GPT rate limit hit, waiting {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  GPT rate limit: giving up after {max_retries} retries")
                    return None, [], None
            else:
                print(f"GPT metadata extraction error: {e}")
                return None, [], None

    return None, [], None


def validate_surnames_with_gpt(surnames: List[str], title: str = None) -> Tuple[List[str], bool]:
    """
    Use GPT to validate if extracted names are valid surnames.

    Args:
        surnames: List of potential surnames to validate
        title: Optional paper title for context

    Returns:
        (validated_surnames, all_valid): List of valid surnames and whether all passed
    """
    if not surnames:
        return [], True

    # Build context
    context = f"Paper title: {title}\n" if title else ""
    context += f"Potential surnames: {', '.join(surnames)}"

    messages = [
        {
            "role": "system",
            "content": """You are validating and correcting author surnames for academic papers.
Return JSON: {"valid_surnames": ["Corrected1", "Corrected2"], "invalid": ["Original1"], "reason": "explanation"}

IMPORTANT: Return CORRECTED surnames in valid_surnames, not the original invalid ones.

Rules:
- Valid surnames are family/last names (e.g., "Smith", "Arkhangelsky", "DeStefano", "O'Brien")
- CORRECT concatenated names: "JonathanRoth" → return "Roth" in valid_surnames
- CORRECT concatenated names: "JonathanRotha" → return "Roth" in valid_surnames (remove trailing 'a')
- REJECT first names with no surname: "Jonathan" alone → invalid, return []
- REJECT title words: "Uncertainty", "Herding", "Citation" → invalid
- REJECT garbled text: "Ndfirm", "Enough" → invalid (unless you can identify the real surname)
- REJECT institution words: "Administrator", "University" → invalid

Examples:
- Input: ["JonathanRotha"] → {"valid_surnames": ["Roth"], "invalid": ["JonathanRotha"], "reason": "extracted surname from concatenated name"}
- Input: ["Smith", "Enough"] → {"valid_surnames": ["Smith"], "invalid": ["Enough"], "reason": "Enough is not a valid surname"}
- Input: ["Dmitry", "Guido"] → {"valid_surnames": [], "invalid": ["Dmitry", "Guido"], "reason": "these are first names, not surnames"}"""
        },
        {
            "role": "user",
            "content": context
        }
    ]

    max_retries = 3
    base_delay = 10.0

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            valid = result.get("valid_surnames", [])
            invalid = result.get("invalid", [])

            # Normalize and filter
            valid = [normalize_case(s) for s in valid if s and isinstance(s, str)]
            valid = [s for s in valid if s and is_valid_surname(s)]

            all_valid = len(invalid) == 0 and len(valid) == len(surnames)

            return valid, all_valid

        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate_limit' in error_str.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  GPT rate limit hit, waiting {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # On rate limit failure, return original with warning
                    return surnames, False
            else:
                print(f"GPT surname validation error: {e}")
                return surnames, False

    return surnames, False


print("GPT functions defined.")


# %%
# Cell 4: PDF Text Search Functions (Fallback for _alert cases)

# Superscript markers that indicate author names
SUPERSCRIPT_MARKERS = r'[¹²³⁴⁵⁶⁷⁸⁹⁰ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ*†‡§¶∗⁺]'
# ORCID icon patterns (may appear as text or special char)
ORCID_PATTERN = r'[\U0001F194]|orcid\.org|ORCID'

def clean_author_markers(text: str) -> str:
    """Remove superscript markers, ORCID icons, and other annotations from author text."""
    # Remove superscript numbers and letters (Unicode)
    text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ]+', '', text)
    # Remove common markers
    text = re.sub(r'[*†‡§¶∗⁺]+', '', text)
    # Remove single-letter superscript markers (a, b, c, 1, 2 etc.) that appear after names
    # Pattern: space + single letter/digit + (comma/end/asterisk/space)
    # e.g., "Roth a,*" -> "Roth", "Smith b " -> "Smith"
    text = re.sub(r'\s+[a-z]\s*[,;*†‡]', '', text)  # "Roth a," -> "Roth"
    text = re.sub(r'\s+[a-z]\s*$', '', text)  # "Roth a" at end -> "Roth"
    text = re.sub(r'\s+\d\s*[,;*†‡]', '', text)  # "Roth 1," -> "Roth"
    text = re.sub(r'\s+\d\s*$', '', text)  # "Roth 1" at end -> "Roth"
    # Remove ORCID references
    text = re.sub(r'\s*\([^)]*orcid[^)]*\)', '', text, flags=re.IGNORECASE)
    # Remove standalone special chars
    text = re.sub(r'\s*[ⓘ🆔]+\s*', ' ', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_surname_from_name(name: str) -> Optional[str]:
    """Extract surname from a full name, handling various formats."""
    name = clean_author_markers(name).strip()
    if not name or len(name) < 2:
        return None

    # Remove Jr., Sr., III, IV, etc.
    name = re.sub(r',?\s*(Jr\.?|Sr\.?|III|IV|II)\s*$', '', name, flags=re.IGNORECASE).strip()

    # Split by spaces
    parts = name.split()
    if not parts:
        return None

    # Last word is surname (unless it's a middle initial)
    surname = parts[-1]

    # If surname looks like initial (single letter or letter with dot), use previous
    if len(surname) <= 2 and len(parts) > 1:
        surname = parts[-2]

    # Normalize case (handle ALL CAPS and Small Caps)
    surname = normalize_case(surname)

    # Remove any remaining punctuation
    surname = re.sub(r'[.,;:\'"]+$', '', surname)

    if surname and is_valid_surname(surname):
        return surname
    return None


def parse_author_line(line: str) -> List[str]:
    """Parse a line that may contain multiple authors separated by various delimiters."""
    authors = []

    # Clean the line
    line = clean_author_markers(line)

    # Split by common separators: comma, semicolon, ampersand, "and", pipe
    # Be careful not to split "Jr." or middle initials
    parts = re.split(r'\s*[,;|]\s*|\s+&\s+|\s+and\s+', line, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Skip if it looks like an affiliation (contains institution keywords)
        affiliation_pattern = r'\b(university|college|institute|school|department|dept|faculty|' \
                             r'administration|administrative|center|centre|laboratory|lab|' \
                             r'research|professor|assistant|associate|fellow|graduate|doctoral|' \
                             r'hospital|clinic|medical|sciences|neurology|medicine)\b'
        if re.search(affiliation_pattern, part, re.IGNORECASE):
            continue

        surname = extract_surname_from_name(part)
        if surname:
            authors.append(surname)

    return authors


def has_superscript_markers(text: str) -> bool:
    """Check if text contains superscript markers that often indicate author names."""
    return bool(re.search(SUPERSCRIPT_MARKERS, text))


def extract_authors_from_text(pdf_path: str) -> Tuple[List[str], Optional[str], bool, bool]:
    """
    Extract authors and year from PDF text using multiple detection patterns.

    Patterns supported:
    - "Author(s):" prefix
    - "By Author Name" prefix
    - Names with superscript markers (¹²³*†‡ᵃᵇᶜ)
    - Vertical list of names (Working Paper style)
    - Horizontal names with separators (comma, and, &, |)
    - Small Caps names (DAVID A. REINSTEIN)
    - Names with Jr./Sr./III suffixes

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

            # === Find year ===
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

            # === Find authors ===

            # Pattern 1: "Author(s):" prefix (explicit field)
            for line in lines[:20]:
                auth_match = re.match(r'^[Aa]uthor\(?s?\)?[:\s]+(.+)$', line)
                if auth_match:
                    authors = parse_author_line(auth_match.group(1))
                    if authors:
                        return authors, year, True, False

            # Pattern 2: "By Author" prefix
            for line in lines[:20]:
                by_match = re.match(r'^[Bb][Yy]\s+(.+)$', line.strip())
                if by_match:
                    authors = parse_author_line(by_match.group(1))
                    if authors:
                        return authors, year, True, False

            # Pattern 3: Lines with superscript markers (strong indicator of author names)
            for line in lines[:25]:
                line = line.strip()
                if 10 < len(line) < 150 and has_superscript_markers(line):
                    # Skip if line looks like title or abstract
                    if re.search(r'^(abstract|introduction|keywords)', line, re.IGNORECASE):
                        continue
                    parsed = parse_author_line(line)
                    if parsed:
                        authors.extend(parsed)

            if authors:
                return list(dict.fromkeys(authors)), year, True, False  # dedupe while preserving order

            # Pattern 4: Vertical list detection (Working Paper style)
            # Look for consecutive short lines that look like names
            name_pattern = re.compile(
                r'^([A-Z][a-zà-ÿ]+)\s+'  # First name
                r'([A-Z]\.?\s+)?'         # Optional middle initial
                r'([A-Z][a-zà-ÿ]+)'       # Last name
                r'[*†‡§¶¹²³ᵃᵇᶜ]*$'        # Optional markers
            )
            consecutive_names = []
            for i, line in enumerate(lines[:30]):
                line = line.strip()
                if 8 < len(line) < 50:
                    # Check if line matches name pattern
                    if name_pattern.match(line):
                        surname = extract_surname_from_name(line)
                        if surname:
                            consecutive_names.append((i, surname))
                    # Also check ALL CAPS pattern
                    elif re.match(r'^[A-Z]{2,}\s+([A-Z]\.?\s+)?[A-Z]{2,}[*†‡§¶¹²³ᵃᵇᶜ]*$', line):
                        surname = extract_surname_from_name(line)
                        if surname:
                            consecutive_names.append((i, surname))

            # If we found consecutive name lines (within 3 lines of each other)
            if len(consecutive_names) >= 2:
                groups = []
                current_group = [consecutive_names[0]]
                for j in range(1, len(consecutive_names)):
                    if consecutive_names[j][0] - consecutive_names[j-1][0] <= 3:
                        current_group.append(consecutive_names[j])
                    else:
                        if len(current_group) >= 2:
                            groups.append(current_group)
                        current_group = [consecutive_names[j]]
                if len(current_group) >= 2:
                    groups.append(current_group)

                # Use the largest group
                if groups:
                    best_group = max(groups, key=len)
                    authors = [name for _, name in best_group]
                    return authors, year, True, False

            # Pattern 5: Single name with markers or standard patterns
            for line in lines[:25]:
                line = line.strip()
                if len(line) > 80 or len(line) < 5:
                    continue

                # Skip common non-author lines
                if re.search(r'^(abstract|introduction|keywords|doi|http|volume|issue)', line, re.IGNORECASE):
                    continue

                # "First M. Last" pattern with optional markers
                match = re.match(
                    r'^([A-Z][a-zà-ÿ]+)\s+'       # First name
                    r'([A-Z]\.?\s+)?'              # Optional middle initial
                    r'([A-Z][a-zà-ÿ]+)'           # Last name
                    r'(?:\s*,\s*(Jr\.?|Sr\.?|III|IV|II))?'  # Optional suffix
                    r'[*†∗‡§¶¹²³ᵃᵇᶜ]*$',          # Optional markers
                    line
                )
                if match and len(line) < 50:
                    surname = match.group(3)
                    if is_valid_surname(surname):
                        authors.append(surname)

                # ALL CAPS / Small Caps pattern
                match = re.match(
                    r'^([A-Z]{2,})\s+'              # First name (caps)
                    r'([A-Z]\.?\s+)?'               # Optional middle
                    r'([A-Z]{2,})'                  # Last name (caps)
                    r'(?:\s*,\s*(Jr\.?|Sr\.?|III|IV|II))?'
                    r'[*†∗‡§¶¹²³ᵃᵇᶜ]*$',
                    line
                )
                if match and len(line) < 45:
                    surname = normalize_case(match.group(3))
                    if surname and is_valid_surname(surname):
                        authors.append(surname)

            # Dedupe while preserving order
            authors = list(dict.fromkeys(authors))

            return authors, year, True, False  # is_readable=True, is_japanese=False

    except Exception as e:
        return [], None, False, False


def extract_authors_from_ocr(pdf_path: str, max_pages: int = 2) -> Tuple[List[str], Optional[str]]:
    """
    Extract authors and year from PDF using OCR.
    Fallback when pdfplumber can't extract text.

    Uses same detection patterns as extract_authors_from_text:
    - "Author(s):" prefix
    - "By Author Name" prefix
    - Names with superscript markers
    - Vertical list of names
    - Horizontal names with separators
    - Small Caps names

    Returns: (authors, year)
    """
    authors = []
    year = None

    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=200)
        if not images:
            return [], None

        for image in images:
            text = pytesseract.image_to_string(image, lang='eng')
            if not text or len(text) < 50:
                continue

            lines = text.split('\n')

            # === Find year ===
            for line in lines[:30]:
                if re.search(r'data|sample|period|survey', line, re.IGNORECASE):
                    continue
                if re.search(r'\b(19[6-9]\d|20[0-2]\d)[–—-](19[6-9]\d|20[0-2]\d)\b', line):
                    continue

                pub_match = re.search(r'(?:published|©|copyright|\(|received|accepted)[:\s]*(\d{4})', line, re.IGNORECASE)
                if pub_match:
                    y = int(pub_match.group(1))
                    if 1960 <= y <= 2026:
                        year = str(y)
                        break

                match = re.search(r'\b(19[6-9]\d|20[0-2]\d)\b', line)
                if match and not year:
                    year = match.group(1)

            # === Find authors ===

            # Pattern 1: "Author(s):" prefix
            for line in lines[:20]:
                auth_match = re.match(r'^[Aa]uthor\(?s?\)?[:\s]+(.+)$', line)
                if auth_match:
                    authors = parse_author_line(auth_match.group(1))
                    if authors:
                        return authors, year

            # Pattern 2: "By Author" prefix
            for line in lines[:20]:
                by_match = re.match(r'^[Bb][Yy]\s+(.+)$', line.strip())
                if by_match:
                    authors = parse_author_line(by_match.group(1))
                    if authors:
                        return authors, year

            # Pattern 3: Lines with superscript markers
            for line in lines[:25]:
                line = line.strip()
                if 10 < len(line) < 150 and has_superscript_markers(line):
                    if re.search(r'^(abstract|introduction|keywords)', line, re.IGNORECASE):
                        continue
                    parsed = parse_author_line(line)
                    if parsed:
                        authors.extend(parsed)

            if authors:
                return list(dict.fromkeys(authors)), year

            # Pattern 4: Vertical list detection
            name_pattern = re.compile(
                r'^([A-Z][a-zà-ÿ]+)\s+'
                r'([A-Z]\.?\s+)?'
                r'([A-Z][a-zà-ÿ]+)'
                r'[*†‡§¶¹²³ᵃᵇᶜ]*$'
            )
            consecutive_names = []
            for i, line in enumerate(lines[:30]):
                line = line.strip()
                if 8 < len(line) < 50:
                    if name_pattern.match(line):
                        surname = extract_surname_from_name(line)
                        if surname:
                            consecutive_names.append((i, surname))
                    elif re.match(r'^[A-Z]{2,}\s+([A-Z]\.?\s+)?[A-Z]{2,}[*†‡§¶¹²³ᵃᵇᶜ]*$', line):
                        surname = extract_surname_from_name(line)
                        if surname:
                            consecutive_names.append((i, surname))

            if len(consecutive_names) >= 2:
                groups = []
                current_group = [consecutive_names[0]]
                for j in range(1, len(consecutive_names)):
                    if consecutive_names[j][0] - consecutive_names[j-1][0] <= 3:
                        current_group.append(consecutive_names[j])
                    else:
                        if len(current_group) >= 2:
                            groups.append(current_group)
                        current_group = [consecutive_names[j]]
                if len(current_group) >= 2:
                    groups.append(current_group)

                if groups:
                    best_group = max(groups, key=len)
                    authors = [name for _, name in best_group]
                    return authors, year

            # Pattern 5: Standard name patterns
            for line in lines[:25]:
                line = line.strip()
                if len(line) > 80 or len(line) < 5:
                    continue

                if re.search(r'^(abstract|introduction|keywords|doi|http|volume|issue)', line, re.IGNORECASE):
                    continue

                # "First M. Last" pattern
                match = re.match(
                    r'^([A-Z][a-zà-ÿ]+)\s+'
                    r'([A-Z]\.?\s+)?'
                    r'([A-Z][a-zà-ÿ]+)'
                    r'(?:\s*,\s*(Jr\.?|Sr\.?|III|IV|II))?'
                    r'[*†∗‡§¶¹²³ᵃᵇᶜ]*$',
                    line
                )
                if match and len(line) < 50:
                    surname = match.group(3)
                    if is_valid_surname(surname):
                        authors.append(surname)

                # ALL CAPS pattern
                match = re.match(
                    r'^([A-Z]{2,})\s+'
                    r'([A-Z]\.?\s+)?'
                    r'([A-Z]{2,})'
                    r'(?:\s*,\s*(Jr\.?|Sr\.?|III|IV|II))?'
                    r'[*†∗‡§¶¹²³ᵃᵇᶜ]*$',
                    line
                )
                if match and len(line) < 45:
                    surname = normalize_case(match.group(3))
                    if surname and is_valid_surname(surname):
                        authors.append(surname)

            if authors:
                authors = list(dict.fromkeys(authors))
                break

        return authors, year

    except Exception as e:
        return [], None


def extract_authors_with_gpt_validation(pdf_path: str, use_gpt: bool = True) -> Tuple[List[str], Optional[str], str]:
    """
    Extract authors from PDF with OCR fallback and GPT validation.

    Flow:
    1. Try pdfplumber text extraction
    2. If fails, try OCR
    3. Validate extracted surnames with GPT
    4. Return validated authors

    Returns: (authors, year, method)
        - method: 'text', 'text_gpt', 'ocr', 'ocr_gpt', or 'none'
    """
    # Try pdfplumber first
    authors, year, is_readable, is_japanese = extract_authors_from_text(pdf_path)

    if is_japanese:
        return [], None, 'japanese'

    method = 'text'

    # If pdfplumber failed, try OCR
    if not is_readable or (not authors and not year):
        ocr_authors, ocr_year = extract_authors_from_ocr(pdf_path)
        if ocr_authors or ocr_year:
            authors = ocr_authors if ocr_authors else authors
            year = ocr_year if ocr_year else year
            method = 'ocr'

    if not authors:
        return [], year, method

    # Validate with GPT if enabled
    if use_gpt and authors:
        validated, all_valid = validate_surnames_with_gpt(authors)
        if validated:
            authors = validated
            method = method + '_gpt'
        # If GPT rejected all (validated is empty), keep original authors

    return authors, year, method


def extract_authors_from_ocr_with_gpt(pdf_path: str, title: str = None) -> Tuple[List[str], Optional[str]]:
    """
    Last-resort author extraction using OCR + GPT.

    This is used for fail files where:
    - Title extraction was wrong → WebSearch failed
    - Pattern-based author extraction also failed

    GPT analyzes the OCR text directly to find author names.

    Returns: (authors, year)
    """
    try:
        # Get OCR text from first 2 pages
        images = convert_from_path(pdf_path, first_page=1, last_page=2, dpi=200)
        if not images:
            return [], None

        ocr_text = ""
        for image in images:
            text = pytesseract.image_to_string(image, lang='eng')
            if text:
                ocr_text += text + "\n"

        if len(ocr_text) < 100:
            return [], None

        # Limit text to first ~3000 chars (focus on header area)
        ocr_text = ocr_text[:3000]

        # Ask GPT to extract authors
        title_context = f'Paper title (if helpful): "{title}"\n\n' if title else ""

        messages = [
            {
                "role": "system",
                "content": """You are extracting author surnames from academic paper OCR text.

Return JSON: {"surnames": ["Surname1", "Surname2"], "year": "YYYY" or null, "confidence": "high/medium/low"}

Rules:
1. Look for the AUTHOR LINE near the top of the paper (usually after title, before abstract)
2. Author names are typically: "FirstName LastName" or "F. LastName" or "FIRSTNAME LASTNAME"
3. Return ONLY surnames (family names), not first names
4. Common patterns:
   - "John Smith, Jane Doe, and Bob Johnson" → ["Smith", "Doe", "Johnson"]
   - "J. Smith¹, J. Doe²" → ["Smith", "Doe"]
   - Names in ALL CAPS: "JOHN SMITH" → ["Smith"]
5. IGNORE:
   - University/institution names
   - Email addresses
   - "Abstract", "Introduction", "Keywords" sections
   - Journal names, copyright text
6. If names look concatenated (no spaces), try to separate them
7. Return empty array if you cannot reliably identify authors
8. Also extract publication year if visible (look for © year, "Published: year", etc.)"""
            },
            {
                "role": "user",
                "content": f"""{title_context}OCR text from PDF (first pages):

{ocr_text}

Extract author surnames and year."""
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
            timeout=30
        )

        result = json.loads(response.choices[0].message.content)
        surnames = result.get('surnames', [])
        year = result.get('year')
        confidence = result.get('confidence', 'low')

        # Validate surnames
        valid_surnames = []
        for name in surnames:
            if not name or len(name) < 2:
                continue
            # Clean and validate
            name = re.sub(r'[^\w\u00C0-\u024F\'-]', '', name)
            if name and is_valid_surname(name):
                valid_surnames.append(normalize_case(name))

        # Validate year
        if year:
            try:
                y = int(year)
                if not (1960 <= y <= 2026):
                    year = None
            except:
                year = None

        return valid_surnames, year

    except Exception as e:
        print(f"    OCR+GPT extraction error: {e}")
        return [], None


print("PDF text search functions defined.")


# %%
# Cell 5: WebSearch Functions
# Primary: Semantic Scholar API
# Fallback: CrossRef API

import requests


def request_with_retry(url: str, params: dict = None, headers: dict = None,
                       timeout: int = 15, max_retries: int = 3,
                       base_delay: float = 2.0) -> Optional[requests.Response]:
    """
    Make HTTP GET request with exponential backoff retry.

    Args:
        url: Request URL
        params: Query parameters
        headers: Request headers
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries (doubles each retry)

    Returns:
        Response object or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                print(f"  Retry {attempt + 1}/{max_retries} after {delay}s... ({type(e).__name__})")
                time.sleep(delay)
            else:
                print(f"  Request failed after {max_retries} retries: {e}")
                return None
        except Exception as e:
            print(f"  Request error: {e}")
            return None
    return None


def search_semantic_scholar(title: str, clean_title: str) -> Tuple[List[str], Optional[str], str]:
    """
    Search Semantic Scholar API with retry logic.
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

        response = request_with_retry(url, params=params, timeout=15, max_retries=3)

        if response and response.status_code == 200:
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
    Search CrossRef API (fallback) with retry logic.
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

        response = request_with_retry(url, params=params, headers=headers, timeout=20, max_retries=3)

        if response and response.status_code == 200:
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
# Cell 6: Step 1 - Extract titles and metadata from PDFs (INCREMENTAL)
# - Loads existing progress and skips already processed files
# - Uses file hash for identification (survives renames)
# - Re-processes failed files
# - Removes entries for deleted files

print(f"Starting extraction at {datetime.now()}")

# Load existing progress
progress = load_progress()

# Auto-remove entries for files in failure/ directory (enables reprocessing)
# When user moves files back from failure/ to ARTICLES_DIR, they will be reprocessed
failure_dir = os.path.join(ARTICLES_DIR, 'failure')
if os.path.exists(failure_dir):
    failure_pdfs = [f for f in os.listdir(failure_dir) if f.lower().endswith('.pdf')]
    if failure_pdfs:
        removed_for_reprocess = 0
        for fname in failure_pdfs:
            fpath = os.path.join(failure_dir, fname)
            fhash = calculate_file_hash(fpath)
            if fhash and fhash in progress['by_hash']:
                del progress['by_hash'][fhash]
                removed_for_reprocess += 1
        if removed_for_reprocess > 0:
            print(f"Removed {removed_for_reprocess} entries for files in failure/ (ready for reprocessing)")
            save_progress(progress)

print(f"Loaded progress: {len(progress['by_hash'])} files previously processed")

# Get current PDF files and calculate hashes
pdf_files = [f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDFs in directory")

# Build hash map for current files
print("Calculating file hashes...")
current_files = {}  # hash -> filename
for filename in pdf_files:
    path = os.path.join(ARTICLES_DIR, filename)
    file_hash = calculate_file_hash(path)
    if file_hash:
        current_files[file_hash] = filename

current_hashes = set(current_files.keys())

# Clean up deleted files
deleted_count = cleanup_deleted_files(progress, current_hashes)
if deleted_count > 0:
    print(f"Removed {deleted_count} entries for deleted files")
    save_progress(progress)

# Determine which files need processing
to_process = []
for file_hash, filename in current_files.items():
    if should_process_extraction(progress, file_hash):
        to_process.append((file_hash, filename))

skipped_count = len(current_files) - len(to_process)
print(f"Files to process: {len(to_process)} (skipping {skipped_count} already processed)")

# Process new/failed files
processed_count = 0
for i, (file_hash, filename) in enumerate(to_process):
    if (i + 1) % 100 == 0:
        print(f"Extracting {i+1}/{len(to_process)}...")

    path = os.path.join(ARTICLES_DIR, filename)

    # Check for Japanese PDF first
    if is_japanese_pdf(path):
        entry = {
            'filename': filename,
            'file_hash': file_hash,
            'title': None,
            'title_method': 'none',
            'title_error': 'japanese',
            'meta_authors': [],
            'meta_year': None,
            'websearch_authors': None,
            'websearch_year': None,
            'websearch_source': None,
            'websearch_status': None,
            'final_authors': None,
            'final_year': None,
            'status': 'japanese',
            'fail_reason': 'japanese',
            'extracted_at': datetime.now().isoformat()
        }
    else:
        title, method, error = extract_title_from_pdf(path)
        meta_authors, meta_year = extract_metadata(path)

        entry = {
            'filename': filename,
            'file_hash': file_hash,
            'title': title,
            'title_method': method,
            'title_error': error,
            'meta_authors': meta_authors,
            'meta_year': meta_year,
            'websearch_authors': None,
            'websearch_year': None,
            'websearch_source': None,
            'websearch_status': None,
            'final_authors': None,
            'final_year': None,
            'status': 'extracted' if title else 'extraction_failed',
            'extracted_at': datetime.now().isoformat()
        }

    # Store in progress
    progress['by_hash'][file_hash] = entry
    progress['hash_to_filename'][file_hash] = filename
    processed_count += 1

    # Save periodically
    if processed_count % SAVE_INTERVAL == 0:
        save_progress(progress)
        print(f"Progress saved at {processed_count}")

# Final save
save_progress(progress)

# Build pdf_data list from progress for subsequent cells
pdf_data = list(progress['by_hash'].values())

# Summary statistics
total_count = len(pdf_data)
japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')
title_count = sum(1 for x in pdf_data if x['title'])
text_count = sum(1 for x in pdf_data if x.get('title_method', '').startswith('text'))
ocr_count = sum(1 for x in pdf_data if x.get('title_method', '').startswith('ocr'))

error_counts = {}
for x in pdf_data:
    err = x.get('title_error')
    if err:
        error_counts[err] = error_counts.get(err, 0) + 1

print(f"\nExtraction complete.")
print(f"Total files in progress: {total_count}")
print(f"  - Newly processed: {processed_count}")
print(f"  - Skipped (already done): {skipped_count}")
print(f"Japanese PDFs (will be moved): {japanese_count}")
print(f"PDFs with title: {title_count}")
print(f"  - From text extraction: {text_count}")
print(f"  - From OCR: {ocr_count}")
print(f"PDFs with metadata authors: {sum(1 for x in pdf_data if x.get('meta_authors'))}")
print(f"PDFs with metadata year: {sum(1 for x in pdf_data if x.get('meta_year'))}")

if error_counts:
    print(f"\nTitle extraction failures:")
    for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  - {err}: {cnt}")

# %%
# Cell 7: Step 2 - WebSearch for PDFs with titles (INCREMENTAL)
# - Skips files that already have WebSearch results
# - Re-tries files where WebSearch previously failed
# - Saves progress to the main progress file

print(f"Starting WebSearch at {datetime.now()}")

# Determine which files need WebSearch
to_search = []
for file_hash, entry in progress['by_hash'].items():
    if should_process_websearch(progress, file_hash):
        to_search.append(file_hash)

already_searched = sum(1 for x in progress['by_hash'].values()
                       if x.get('websearch_status') in ('done', 'not_found'))
japanese_count = sum(1 for x in progress['by_hash'].values() if x.get('status') == 'japanese')
no_title_count = sum(1 for x in progress['by_hash'].values()
                     if x.get('status') != 'japanese' and not x.get('title'))

print(f"Files to search: {len(to_search)}")
print(f"Already searched: {already_searched}")
print(f"Skipping: {japanese_count} Japanese, {no_title_count} no title")

# Process WebSearch
search_count = 0
for i, file_hash in enumerate(to_search):
    entry = progress['by_hash'][file_hash]

    if (i + 1) % 10 == 0:
        print(f"Searching {i+1}/{len(to_search)}... ({entry['filename'][:30]})")

    # Perform WebSearch (Semantic Scholar -> CrossRef fallback)
    ws_authors, ws_year, ws_source = websearch_authors(entry['title'])

    entry['websearch_authors'] = ws_authors if ws_authors else []
    entry['websearch_year'] = ws_year
    entry['websearch_source'] = ws_source if ws_source else None
    entry['websearch_status'] = 'done' if ws_authors else 'not_found'
    entry['websearch_at'] = datetime.now().isoformat()

    search_count += 1

    # Save progress periodically
    if search_count % SAVE_INTERVAL == 0:
        save_progress(progress)
        print(f"Progress saved at {search_count}")

# Final save
save_progress(progress)

# Rebuild pdf_data from progress
pdf_data = list(progress['by_hash'].values())

# WebSearch statistics
ws_success = sum(1 for x in pdf_data if x.get('websearch_authors'))
ws_semantic = sum(1 for x in pdf_data if x.get('websearch_source') == 'semantic_scholar')
ws_crossref = sum(1 for x in pdf_data if x.get('websearch_source') == 'crossref')
ws_no_title = sum(1 for x in pdf_data if x.get('status') != 'japanese' and not x.get('title'))
ws_not_found = sum(1 for x in pdf_data if x.get('status') != 'japanese' and x.get('title')
                   and not x.get('websearch_authors'))

print(f"\nWebSearch complete at {datetime.now()}")
print(f"  - Newly searched: {search_count}")
print(f"Files with WebSearch results: {ws_success}")
print(f"  - From Semantic Scholar: {ws_semantic}")
print(f"  - From CrossRef: {ws_crossref}")
print(f"Files without WebSearch results: {ws_no_title + ws_not_found}")
print(f"  - No title extracted: {ws_no_title}")
print(f"  - Title found but search failed: {ws_not_found}")

# %%
# Cell 8: Step 3 - Compare WebSearch results with metadata and determine final authors/year
# Now includes GPT validation for WebSearch results to catch API errors (e.g., "Enough" instead of "Englich")
# Skips already processed files (incremental processing)

print("Comparing WebSearch results with metadata (with GPT validation)...")

GPT_VALIDATE_WEBSEARCH = True  # Set to False to skip GPT validation (faster but less accurate)
gpt_validated_count = 0
gpt_corrected_count = 0
skipped_count = 0

for i, item in enumerate(pdf_data):
    if (i + 1) % 100 == 0:
        print(f"Processing {i+1}/{len(pdf_data)}...")

    # Skip if already successfully processed with GPT validation
    if item.get('status') == 'success' and item.get('gpt_validation') in ('passed', 'corrected'):
        skipped_count += 1
        continue

    filename = item['filename']
    ws_authors = item['websearch_authors'] or []
    ws_year = item['websearch_year']
    meta_authors = item['meta_authors'] or []
    meta_year = item['meta_year']
    title = item.get('title', '')

    # Filter valid surnames (initial filter)
    ws_valid = [normalize_case(a) for a in ws_authors if is_valid_surname(a)]
    meta_valid = [normalize_case(a) for a in meta_authors if is_valid_surname(a)]

    # GPT validation for WebSearch results
    if GPT_VALIDATE_WEBSEARCH and ws_valid:
        validated, all_valid = validate_surnames_with_gpt(ws_valid, title)
        gpt_validated_count += 1
        if not all_valid or validated != ws_valid:
            gpt_corrected_count += 1
            item['gpt_validation'] = 'corrected'
            item['original_ws_authors'] = ws_valid
        else:
            item['gpt_validation'] = 'passed'
        ws_valid = validated

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
print(f"  Skipped (already processed): {skipped_count}")
if GPT_VALIDATE_WEBSEARCH:
    print(f"\nGPT Validation (WebSearch results):")
    print(f"  Validated: {gpt_validated_count}")
    print(f"  Corrected: {gpt_corrected_count}")

# %%
# Cell 9: Step 4 - PDF Text Search for _alert files (with OCR fallback and GPT validation)
# Skips already processed files (incremental processing)

# Only process files that need PDF text search AND haven't been GPT validated yet
alert_files = [x for x in pdf_data
               if x['status'] == 'alert'
               and x.get('pdf_gpt_validation') not in ('passed', 'corrected')]
already_processed = sum(1 for x in pdf_data
                        if x.get('pdf_gpt_validation') in ('passed', 'corrected'))

print(f"Processing {len(alert_files)} alert files with PDF text search...")
print(f"Skipping {already_processed} already GPT-validated files")
print("Using: pdfplumber -> OCR fallback -> GPT surname validation")

GPT_VALIDATE_PDF_TEXT = True  # Set to False to skip GPT validation for PDF text extraction
pdf_gpt_validated = 0
pdf_gpt_corrected = 0

for i, item in enumerate(alert_files):
    if (i + 1) % 10 == 0:
        print(f"Processing {i+1}/{len(alert_files)}...")

    path = os.path.join(ARTICLES_DIR, item['filename'])
    title = item.get('title', '')

    # Extract authors with OCR fallback and GPT validation
    # This function: pdfplumber -> OCR if needed -> GPT validates surnames
    text_authors, text_year, method = extract_authors_with_gpt_validation(path, use_gpt=True)

    # Additional GPT validation for extracted authors (catch concatenated names, etc.)
    if GPT_VALIDATE_PDF_TEXT and text_authors:
        validated, all_valid = validate_surnames_with_gpt(text_authors, title)
        pdf_gpt_validated += 1
        if not all_valid or validated != text_authors:
            pdf_gpt_corrected += 1
            item['pdf_gpt_validation'] = 'corrected'
            item['original_pdf_authors'] = text_authors
        else:
            item['pdf_gpt_validation'] = 'passed'
        text_authors = validated

    # Update authors if we found valid ones
    if text_authors and not item['final_authors']:
        item['final_authors'] = text_authors
        item['author_source'] = f'pdf_{method}'

    # Update year if we found one
    if text_year and not item['final_year']:
        item['final_year'] = text_year
        item['year_source'] = f'pdf_{method.split("_")[0]}'  # 'pdf_text' or 'pdf_ocr'

    # Check if we now have valid data
    if item['final_authors'] and item['final_year']:
        item['status'] = 'success'
    elif method == 'japanese':
        item['status'] = 'japanese'
        item['fail_reason'] = 'japanese'
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
print(f"  Alert (remaining): {alert_count}")
print(f"  Japanese: {japanese_count}")
if GPT_VALIDATE_PDF_TEXT:
    print(f"\nGPT Validation (PDF text extraction):")
    print(f"  Validated: {pdf_gpt_validated}")
    print(f"  Corrected: {pdf_gpt_corrected}")


# %%
# Cell 9.5: Last-resort OCR+GPT extraction for fail files
# - Targets files with status='fail' and fail_reason='no_valid_authors'
# - Uses OCR + GPT to extract authors directly from PDF images
# - If successful, changes status to 'alert' (requires human verification)
# - If still fails, keeps status='fail'

OCR_GPT_FALLBACK = True  # Set to False to skip this step

fail_files = [x for x in pdf_data
              if x['status'] == 'fail'
              and x.get('fail_reason') == 'no_valid_authors'
              and x.get('ocr_gpt_attempted') != True]

print(f"\n=== Cell 9.5: Last-resort OCR+GPT extraction ===")
print(f"Files to process: {len(fail_files)} (fail with no_valid_authors)")

if OCR_GPT_FALLBACK and fail_files:
    ocr_gpt_success = 0
    ocr_gpt_fail = 0

    for i, item in enumerate(fail_files):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(fail_files)}...")

        filename = item['filename']
        file_hash = item.get('file_hash', '')

        # Find current file path (may have been renamed)
        current_filename = None
        for f in os.listdir(ARTICLES_DIR):
            if f.lower().endswith('.pdf'):
                path = os.path.join(ARTICLES_DIR, f)
                if calculate_file_hash(path) == file_hash:
                    current_filename = f
                    break

        if not current_filename:
            print(f"  File not found: {filename}")
            item['ocr_gpt_attempted'] = True
            ocr_gpt_fail += 1
            continue

        pdf_path = os.path.join(ARTICLES_DIR, current_filename)
        title = item.get('title')

        # Extract authors using OCR + GPT
        authors, year = extract_authors_from_ocr_with_gpt(pdf_path, title)

        item['ocr_gpt_attempted'] = True
        item['ocr_gpt_authors'] = authors
        item['ocr_gpt_year'] = year

        if authors:
            # Success! Change to alert status
            item['final_authors'] = authors
            item['author_source'] = 'ocr_gpt_fallback'
            if year and not item.get('final_year'):
                item['final_year'] = year
                item['year_source'] = 'ocr_gpt_fallback'

            # Check if we now have valid data
            if item['final_authors'] and item.get('final_year'):
                item['status'] = 'alert'
                item['alert_reason'] = 'ocr_gpt_fallback'
                del item['fail_reason']
                ocr_gpt_success += 1
                print(f"  Recovered: {current_filename} → {authors}")
            elif item['final_authors']:
                # Have authors but no year - still alert (can use n.d.)
                item['status'] = 'alert'
                item['alert_reason'] = 'ocr_gpt_fallback_no_year'
                del item['fail_reason']
                ocr_gpt_success += 1
                print(f"  Recovered (no year): {current_filename} → {authors}")
            else:
                ocr_gpt_fail += 1
        else:
            ocr_gpt_fail += 1

    print(f"\nOCR+GPT fallback results:")
    print(f"  Recovered to alert: {ocr_gpt_success}")
    print(f"  Still fail: {ocr_gpt_fail}")

    # Update progress
    for item in pdf_data:
        file_hash = item.get('file_hash', '')
        if file_hash and file_hash in progress['by_hash']:
            progress['by_hash'][file_hash].update({
                'status': item['status'],
                'final_authors': item.get('final_authors'),
                'final_year': item.get('final_year'),
                'author_source': item.get('author_source'),
                'ocr_gpt_attempted': item.get('ocr_gpt_attempted'),
                'ocr_gpt_authors': item.get('ocr_gpt_authors'),
                'alert_reason': item.get('alert_reason'),
                'fail_reason': item.get('fail_reason')
            })
    save_progress(progress)

# Final counts after OCR+GPT fallback
success_count = sum(1 for x in pdf_data if x['status'] == 'success')
fail_count = sum(1 for x in pdf_data if x['status'] == 'fail')
alert_count = sum(1 for x in pdf_data if x['status'] == 'alert')
japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')

print(f"\nFinal status after all extractions:")
print(f"  Success: {success_count}")
print(f"  Alert: {alert_count}")
print(f"  Fail: {fail_count}")
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
    elif item['status'] == 'alert':
        # Alert: generate filename and move to alert/ directory for human review
        new_name = generate_filename(item['final_authors'], item['final_year'], existing)
        if new_name:
            item['new_filename'] = new_name
            existing.add(new_name.lower())
        else:
            item['new_filename'] = None
        item['move_to_alert'] = True
    elif item['status'] == 'fail':
        # Fail: no filename, move to failure/ directory
        item['new_filename'] = None
        item['move_to_failure'] = True
    # Japanese files keep their original name and move to japanese/

success_count = sum(1 for x in pdf_data if x['status'] == 'success' and x.get('new_filename'))
alert_count = sum(1 for x in pdf_data if x.get('move_to_alert'))
failure_count = sum(1 for x in pdf_data if x.get('move_to_failure'))
japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')

print(f"Filename generation complete.")
print(f"  Files to rename: {success_count}")
print(f"  Files to move to alert/: {alert_count}")
print(f"  Files to move to failure/: {failure_count}")
print(f"  Files to move to japanese/: {japanese_count}")

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

EXECUTE_RENAME = True  # Set to True to actually rename files

if EXECUTE_RENAME:
    print("Starting file rename...")

    # Create folders if needed
    japanese_dir = os.path.join(ARTICLES_DIR, 'japanese')
    research_dir = os.path.join(ARTICLES_DIR, 're-search')
    alert_dir = os.path.join(ARTICLES_DIR, 'alert')
    failure_dir = os.path.join(ARTICLES_DIR, 'failure')
    for folder in [japanese_dir, research_dir, alert_dir, failure_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    renamed = []
    moved_japanese = []
    moved_research = []
    moved_alert = []
    moved_failure = []
    errors = []

    for item in pdf_data:
        old_name = item['filename']
        old_path = os.path.join(ARTICLES_DIR, old_name)

        # Skip if file doesn't exist (already moved or deleted)
        if not os.path.exists(old_path):
            continue

        # Handle Japanese papers - move to japanese folder
        if item['status'] == 'japanese':
            new_path = os.path.join(japanese_dir, old_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_japanese.append({'old': old_name, 'new': f'japanese/{old_name}'})
                else:
                    errors.append({'old': old_name, 'new': f'japanese/{old_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f'japanese/{old_name}', 'reason': str(e)})
            continue

        # Handle alert papers - rename and move to alert folder (for human review)
        if item.get('move_to_alert'):
            new_name = item.get('new_filename') or old_name
            new_path = os.path.join(alert_dir, new_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_alert.append({'old': old_name, 'new': f'alert/{new_name}',
                                       'alert_reason': item.get('alert_reason'),
                                       'authors': item.get('final_authors')})
                else:
                    errors.append({'old': old_name, 'new': f'alert/{new_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f'alert/{new_name}', 'reason': str(e)})
            continue

        # Handle fail papers - move to failure folder (no rename)
        if item.get('move_to_failure'):
            new_path = os.path.join(failure_dir, old_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_failure.append({'old': old_name, 'new': f'failure/{old_name}',
                                         'fail_reason': item.get('fail_reason')})
                else:
                    errors.append({'old': old_name, 'new': f'failure/{old_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f'failure/{old_name}', 'reason': str(e)})
            continue

        # Handle OCR-only papers - move to re-search folder for manual verification
        if item.get('title_method') == 'ocr' and item.get('new_filename'):
            new_name = item['new_filename']
            new_path = os.path.join(research_dir, new_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_research.append({'old': old_name, 'new': f're-search/{new_name}'})
                else:
                    errors.append({'old': old_name, 'new': f're-search/{new_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f're-search/{new_name}', 'reason': str(e)})
            continue

        # Handle regular renaming
        new_name = item.get('new_filename')

        if not new_name or old_name == new_name:
            continue

        new_path = os.path.join(ARTICLES_DIR, new_name)

        try:
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                renamed.append({'old': old_name, 'new': new_name})
            else:
                errors.append({'old': old_name, 'new': new_name, 'reason': 'path conflict'})
        except Exception as e:
            errors.append({'old': old_name, 'new': new_name, 'reason': str(e)})

    print(f"\nRename complete:")
    print(f"  Renamed: {len(renamed)}")
    print(f"  Moved to japanese/: {len(moved_japanese)}")
    print(f"  Moved to alert/: {len(moved_alert)}")
    print(f"  Moved to failure/: {len(moved_failure)}")
    print(f"  Moved to re-search/: {len(moved_research)}")
    print(f"  Errors: {len(errors)}")

    # Show alert details
    if moved_alert:
        print(f"\n--- Alert files (require human review) ---")
        for item in moved_alert[:20]:
            print(f"  {item['old']} -> {item['new']}")
            print(f"    Reason: {item.get('alert_reason')}, Authors: {item.get('authors')}")

    # Save results
    with open(os.path.join(ARTICLES_DIR, 'rename_results_final.json'), 'w') as f:
        json.dump({
            'renamed': renamed,
            'moved_japanese': moved_japanese,
            'moved_alert': moved_alert,
            'moved_failure': moved_failure,
            'moved_research': moved_research,
            'errors': errors,
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

    # Show preview of OCR-only files
    ocr_files = [x for x in pdf_data if x.get('title_method') == 'ocr']
    if ocr_files:
        print(f"\nOCR-only files to be moved to re-search/ ({len(ocr_files)}):")
        for item in ocr_files[:10]:
            print(f"  {item['filename']} -> re-search/")

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
    'title_extraction': {
        'text': sum(1 for x in pdf_data if x.get('title_method', '').startswith('text')),
        'ocr': sum(1 for x in pdf_data if x.get('title_method', '').startswith('ocr')),
        'none': sum(1 for x in pdf_data if x.get('title_method') == 'none'),
    },
    'title_errors': {},
    'author_sources': {
        'websearch': sum(1 for x in pdf_data if x.get('author_source') == 'websearch'),
        'metadata': sum(1 for x in pdf_data if x.get('author_source') == 'metadata'),
        'pdf_text': sum(1 for x in pdf_data if x.get('author_source') == 'pdf_text'),
        'none': sum(1 for x in pdf_data if x.get('author_source') == 'none'),
    },
    'websearch_sources': {
        'semantic_scholar': sum(1 for x in pdf_data if x.get('websearch_source') == 'semantic_scholar'),
        'crossref': sum(1 for x in pdf_data if x.get('websearch_source') == 'crossref'),
        'no_title': sum(1 for x in pdf_data if x['status'] != 'japanese' and not x['title']),
        'not_found': sum(1 for x in pdf_data if x['status'] != 'japanese' and x['title'] and not x.get('websearch_authors')),
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

# Count title extraction errors
for item in pdf_data:
    err = item.get('title_error')
    if err:
        summary['title_errors'][err] = summary['title_errors'].get(err, 0) + 1

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
