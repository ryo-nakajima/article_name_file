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
    progress_path = os.path.join(DATA_DIR, PROGRESS_FILE)
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
    progress_path = os.path.join(DATA_DIR, PROGRESS_FILE)
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
    """Check if file needs WebSearch (extracted but not searched, or previously failed/alert)."""
    if file_hash not in progress['by_hash']:
        return False  # Can't search without extraction

    entry = progress['by_hash'][file_hash]

    # Skip Japanese files
    if entry.get('status') == 'japanese':
        return False

    # Skip if no title extracted
    if not entry.get('title'):
        return False

    # Re-process files that previously failed or were alert
    # This allows new fixes (filename parsing, title normalization) to be applied
    status = entry.get('status', '')
    if status in ('fail', 'alert'):
        return True  # Re-process failed/alert files

    # Re-process files with journal-like suffix after year (e.g., Author_2020_AER.pdf)
    # Skip single-letter suffixes (_a, _b, _c) which are for disambiguation
    filename = entry.get('filename', '')
    if filename.lower().endswith('.pdf'):
        base = filename[:-4]
        parts = base.split('_')
        if len(parts) >= 2:
            # Find year position
            for i, part in enumerate(parts):
                if re.match(r'^\d{4}$', part) and i < len(parts) - 1:
                    # Year is not the last part - check suffix length
                    suffix = '_'.join(parts[i+1:])
                    # Only reprocess if suffix is 2+ characters (journal abbreviation)
                    # Skip single lowercase letter (disambiguation suffix like _a, _b)
                    if not (len(suffix) == 1 and suffix.islower()):
                        return True
                    break

    # Process if websearch not done
    websearch_status = entry.get('websearch_status', '')
    if websearch_status in ('done', 'not_found'):
        return False  # Already searched and succeeded

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

# Directory paths
# When running from code/, ARTICLES_DIR is the parent directory
CODE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
ARTICLES_DIR = os.path.dirname(CODE_DIR)  # Parent of code/
DATA_DIR = os.path.join(CODE_DIR, 'data')  # code/data/ for JSON files

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print(f"Code directory: {CODE_DIR}")
print(f"Articles directory: {ARTICLES_DIR}")
print(f"Data directory: {DATA_DIR}")
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

    # Length check: 2-20 chars (real surnames rarely exceed 15)
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

    lower_name = name.lower()

    # Reject concatenated title/English words (e.g., "Evidencefromradiologists")
    # These have common English words embedded without spaces
    if len(name) > 10:
        embedded_words = ['from', 'with', 'the', 'and', 'for', 'evidence', 'analysis',
                          'study', 'effect', 'model', 'data', 'using', 'based',
                          'market', 'price', 'labor', 'trade', 'policy', 'average',
                          'treatment', 'selection', 'patent', 'innovation', 'court',
                          'experiment', 'estimat', 'causal', 'empiric']
        for word in embedded_words:
            idx = lower_name.find(word)
            if idx > 0:  # Word is embedded (not at start)
                return False

    # Reject garbled/truncated names: unusual starting consonant clusters
    # Be careful not to reject valid names like "McCrary", "Schwarzenegger", "Ng", "Ndegwa"
    if len(lower_name) >= 3:
        # Check for clearly invalid 3+ consonant clusters at start
        invalid_starts = ['ndf', 'xzx', 'zxz', 'qxq', 'vbv', 'bvb', 'kmt', 'pmt', 'bnt', 'dnt', 'fnt', 'gnt', 'hnt', 'lnt', 'mnt', 'pnt', 'rnt', 'tnt', 'wnt', 'znt']
        for inv in invalid_starts:
            if lower_name.startswith(inv):
                return False
        # Relaxed: reject only if starts with 5+ consonants (was 4)
        # This allows: McCr, Schw, Strz, etc.
        if re.match(r'^[bcdfghjklmnpqrstvwxz]{5,}', lower_name):
            return False

    # Check minimum vowel ratio for longer names (names need vowels!)
    # Relaxed: 10% threshold (was 15%) to allow names like "Thursby" (1/7 = 14%)
    if len(lower_name) >= 6:  # Only check for longer names
        vowels = sum(1 for c in lower_name if c in 'aeiou')
        if vowels < len(lower_name) * 0.10:  # Less than 10% vowels is suspicious
            return False

    # Handle lowercase prefixes (de, d', van, von, di, la, le, el, al, etc.)
    # These are valid surname prefixes in many cultures
    lowercase_prefixes = ['de ', "d'", 'van ', 'von ', 'di ', 'da ', 'la ', 'le ', 'el ', 'al ', 'del ', 'della ', 'lo ', 'den ', 'ter ', 'ten ']
    starts_with_valid_prefix = any(name.lower().startswith(p) for p in lowercase_prefixes)

    # Reject if starts with lowercase, UNLESS it's a valid prefix
    if name and name[0].islower() and not starts_with_valid_prefix:
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


def parse_existing_filename(filename: str) -> Tuple[List[str], Optional[str], bool]:
    """
    Parse existing filename to extract authors and year if it follows academic naming convention.

    Patterns recognized:
    - Author_Year.pdf
    - Author_Author_Year.pdf
    - Author_et_al_Year.pdf
    - Author_Year_a.pdf (with suffix)
    - Author_Year_AER.pdf (with journal abbreviation suffix)
    - Author_et_al_Year_QJE.pdf

    Returns:
        (authors, year, is_valid_format)
    """
    if not filename.lower().endswith('.pdf'):
        return [], None, False

    base = filename[:-4]  # Remove .pdf

    # Split by underscore
    parts = base.split('_')
    if len(parts) < 2:
        return [], None, False

    # Find the position of a 4-digit year in the parts
    year_index = None
    for i, part in enumerate(parts):
        if re.match(r'^\d{4}$', part):
            year_index = i
            break  # Use the first year found

    if year_index is None:
        return [], None, False

    year = parts[year_index]
    author_parts = parts[:year_index]  # Everything before the year

    if not author_parts:
        return [], None, False

    # Check for et_al pattern
    if len(author_parts) >= 3 and author_parts[-2].lower() == 'et' and author_parts[-1].lower() == 'al':
        # Author_et_al_Year pattern
        author = author_parts[0]
        if is_valid_surname(author):
            return [author], year, True
        return [], None, False

    # Regular pattern: Author(_Author)*_Year
    valid_authors = []
    for a in author_parts:
        # Skip 'et' and 'al' as they're part of et_al pattern
        if a.lower() in ('et', 'al'):
            continue
        if is_valid_surname(a):
            valid_authors.append(a)

    if valid_authors:
        return valid_authors, year, True

    return [], None, False


# ============================================================================
# BOILERPLATE DETECTION - Used to reject JSTOR/publisher copyright text
# ============================================================================

# Common boilerplate patterns that should NEVER appear in titles
BOILERPLATE_INDICATORS = [
    'collaborating with jstor',
    'jstor is a not-for-profit',
    'american economic association is collaborating',
    'your use of the jstor archive',
    'terms and conditions',
    'this content downloaded',
    'all use subject to',
    'accessed:',
    'please contact jstor',
    'digitize, preserve and extend',
    'trusted digital archive',
    'stable url:',
    'linked references are available',
    'for more information about jstor',
]

# Patterns that indicate JSTOR cover page structure
JSTOR_COVER_INDICATORS = [
    'jstor is a not-for-profit',
    'your use of the jstor archive',
    'this content downloaded from',
]


def is_boilerplate_text(text: str) -> bool:
    """Check if text is boilerplate/copyright text that should be rejected as title."""
    if not text:
        return False
    text_lower = text.lower()
    return any(ind in text_lower for ind in BOILERPLATE_INDICATORS)


def is_jstor_cover_page(text: str) -> bool:
    """Check if the page is a JSTOR cover page with structured metadata."""
    if not text:
        return False
    text_lower = text.lower()
    return any(ind in text_lower for ind in JSTOR_COVER_INDICATORS)


def extract_title_from_jstor_cover(text: str) -> Optional[str]:
    """
    Extract title from JSTOR cover page using its known structure.

    JSTOR cover page structure (common patterns):
    Pattern A: Publisher name first
    - Line 0: Publisher name (e.g., "American Economic Association")
    - Line 1+: Title (may span multiple lines)
    - "Author(s):" line marks end of title

    Pattern B: Title first
    - Line 0: Title (may span multiple lines)
    - "Author(s):" line marks end of title

    Returns:
        Extracted title or None if extraction fails
    """
    lines = text.split('\n')

    title_lines = []
    started = False

    # Known publisher/journal names to skip
    publisher_patterns = [
        'american economic association',
        'econometric society',
        'quarterly journal of economics',
        'journal of political economy',
        'review of economic studies',
        'economic journal',
        'national bureau of economic research',
        'rand journal of economics',
        'journal of labor economics',
        'journal of finance',
        'journal of monetary economics',
        'journal of econometrics',
        'oxford university press',
        'cambridge university press',
        'wiley',
        ', ltd.',
        ', inc.',
    ]

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip first non-empty line IF it looks like a publisher name
        if not started:
            line_lower = line.lower()
            # Check if it's a known publisher
            is_publisher = any(pub in line_lower for pub in publisher_patterns)

            # Only skip if it's a KNOWN publisher name
            # Don't skip generic short lines - they might be titles
            if is_publisher and i == 0:
                started = True
                continue
            started = True

        # Stop at Author(s) line
        if line.lower().startswith('author(s)') or line.lower().startswith('authors:'):
            break

        # Stop at Source line
        if line.lower().startswith('source:'):
            break

        # Stop at URL patterns
        if line.lower().startswith('stable url') or line.lower().startswith('http'):
            break

        # Stop at boilerplate
        if is_boilerplate_text(line):
            break

        # Stop at "Published by" line
        if line.lower().startswith('published by'):
            break

        # Add to title if it looks like title text
        if len(line) > 3:
            title_lines.append(line)

        # Limit title to 5 lines max
        if len(title_lines) >= 5:
            break

    if title_lines:
        title = ' '.join(title_lines)
        # Clean up
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove trailing special characters
        title = re.sub(r'[\d\*†‡§¶]+$', '', title).strip()

        # Validate: reasonable length, not boilerplate
        if 15 <= len(title) <= 300 and not is_boilerplate_text(title):
            return title

    return None


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

            # CRITICAL: Skip boilerplate/copyright text (JSTOR, etc.)
            if is_boilerplate_text(text):
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
                    # === JSTOR COVER PAGE HANDLING ===
                    # JSTOR PDFs have a structured cover page - extract title specially
                    if is_jstor_cover_page(text):
                        jstor_title = extract_title_from_jstor_cover(text)
                        if jstor_title and not is_boilerplate_text(jstor_title):
                            method = 'text_jstor'
                            return jstor_title, method, None

                    # Try font/position-based detection with GPT validation
                    title = find_title_with_gpt_validation(page, text, use_gpt=use_gpt)
                    if title and not is_boilerplate_text(title):
                        method = 'text_gpt' if use_gpt else 'text'
                        return title, method, None

                    # Fallback to simple line-based detection (without GPT for speed)
                    lines = text.split('\n')
                    for line in lines[:25]:
                        line = line.strip()
                        if len(line) < 15:
                            continue

                        # Skip boilerplate text
                        if is_boilerplate_text(line):
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
                            # Final boilerplate check before returning
                            if len(title) >= 15 and not is_boilerplate_text(title):
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

    # Skip copyright/boilerplate pages (JSTOR, NBER, etc.)
    # These often appear at the start of scanned PDFs
    boilerplate_indicators = [
        'collaborating with jstor',
        'jstor is a not-for-profit',
        'accessed:',
        'your use of the jstor archive',
        'terms and conditions',
        'please contact jstor',
        'this content downloaded',
        'all use subject to',
        'nber working paper',
        'working paper no.',
        'discussion paper no.'
    ]

    text_lower = text[:2000].lower()
    is_boilerplate = any(ind in text_lower for ind in boilerplate_indicators)

    if is_boilerplate:
        # Try to find actual content after boilerplate (look for double newlines)
        # Often the real content starts after the first page
        parts = text.split('\n\n\n')
        if len(parts) > 1:
            # Skip the first part (boilerplate) and use the rest
            text = '\n\n'.join(parts[1:])
        else:
            # Try splitting by page markers or common patterns
            for marker in ['Abstract', 'ABSTRACT', 'Introduction', 'INTRODUCTION', '1.', '1 ']:
                if marker in text:
                    idx = text.index(marker)
                    if idx > 100:  # Make sure we're not at the very beginning
                        # Look for title before the marker
                        pre_text = text[:idx]
                        # Find the last substantial paragraph before Abstract
                        lines = [l.strip() for l in pre_text.split('\n') if l.strip()]
                        # Take last few significant lines as potential title area
                        title_candidates = [l for l in lines[-10:] if len(l) > 20 and not any(bi in l.lower() for bi in boilerplate_indicators)]
                        if title_candidates:
                            text = '\n'.join(title_candidates) + '\n' + text[idx:idx+500]
                            break

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
- IGNORE copyright notices, JSTOR collaboration text, terms of use, etc.
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

    # Reject if title looks like boilerplate
    title_lower = title.lower()
    if any(ind in title_lower for ind in boilerplate_indicators):
        return None

    return title if len(title) >= 15 else None


def normalize_title_with_gpt(raw_title: str) -> Optional[str]:
    """
    Use GPT to normalize a garbled/concatenated title into proper format.

    Handles:
    - Missing spaces: "TAXATIONANDINNOVATION" -> "Taxation and Innovation"
    - Mixed case issues: "AnAnalysisOfMarkets" -> "An Analysis of Markets"
    - Special characters: "∗†‡" removal
    - Author names embedded in title extraction

    Args:
        raw_title: The raw/garbled title text

    Returns:
        Normalized title or None if cannot be normalized
    """
    if not raw_title or len(raw_title) < 10:
        return None

    # Skip if already looks clean (has normal spacing)
    words = raw_title.split()
    if len(words) >= 3 and all(len(w) < 25 for w in words):
        # Already has reasonable word structure
        return raw_title

    messages = [
        {
            "role": "system",
            "content": """You are an expert at reconstructing academic paper titles from garbled text.

The input may have:
- Missing spaces between words (e.g., "TAXATIONANDINNOVATION" -> "Taxation and Innovation")
- Concatenated author names at the end (e.g., "...CENTURY U A FUK KCIGIT" -> remove author parts)
- Special symbols (∗, †, ‡, §) that should be removed
- ALL CAPS that should be converted to Title Case

Your task:
1. Insert spaces where words were concatenated
2. Remove any author names that got merged with the title
3. Remove special symbols and footnote markers
4. Return ONLY the reconstructed paper title in proper Title Case
5. If you cannot determine a valid title, return "INVALID"

Examples:
Input: "TAXATIONANDINNOVATIONINTHE ∗ TWENTIETHCENTURY U A FUK KCIGIT"
Output: "Taxation and Innovation in the Twentieth Century"

Input: "ESTIMATINGDYNAMICDISCRETECHOICEMODELSWITHHYPERBOLIC"
Output: "Estimating Dynamic Discrete Choice Models with Hyperbolic Discounting"
"""
        },
        {
            "role": "user",
            "content": f"Reconstruct this title:\n{raw_title[:500]}"
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()

        if result == "INVALID" or len(result) < 10:
            return None

        # Clean up result
        result = re.sub(r'^["\'"]|["\'"]$', '', result)
        result = re.sub(r'[\*†‡§¶]+', '', result).strip()

        return result if len(result) >= 15 else None

    except Exception as e:
        print(f"    GPT title normalization error: {e}")
        return None


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


def validate_surnames_with_gpt(surnames: List[str], title: str = None, trust_source: bool = True) -> Tuple[List[str], bool]:
    """
    Use GPT to validate if extracted names are valid surnames.

    Args:
        surnames: List of potential surnames to validate
        title: Optional paper title for context
        trust_source: If True (default), fallback to original if GPT returns empty.
                      Set to False for untrusted sources (metadata, OCR) to be stricter.

    Returns:
        (validated_surnames, all_valid): List of valid surnames and whether all passed
    """
    if not surnames:
        return [], True

    # Pre-filter: keep only names that pass basic validation
    original_valid = [s for s in surnames if is_valid_surname(s)]
    if not original_valid:
        return [], True  # Nothing to validate

    # Build context
    context = f"Paper title: {title}\n" if title else ""
    context += f"Potential surnames: {', '.join(surnames)}"

    messages = [
        {
            "role": "system",
            "content": """You are validating author surnames for academic papers.
Return JSON: {"valid_surnames": ["Name1", "Name2"], "invalid": ["Bad1"], "reason": "explanation"}

CRITICAL - REJECT these patterns:
1. CONCATENATED TITLE WORDS: Words from paper titles joined without spaces
   - "Evidencefromradiologists" → INVALID (title fragment "Evidence from radiologists")
   - "Evidencefromthecourts" → INVALID (title fragment "Evidence from the courts")
   - "EstimatingAverage" → INVALID (title fragment "Estimating Average")
   - "Patentsandcumulative" → INVALID (title fragment)
   - "Selectionwithvariation" → INVALID (title fragment)

2. EMBEDDED ENGLISH WORDS: Names containing common words in the middle
   - Any name with "from", "with", "the", "and" embedded → INVALID
   - Any name with "evidence", "effect", "model", "market", "policy" embedded → INVALID

3. OBVIOUS NON-SURNAMES:
   - First names alone: "Jonathan", "David" → INVALID
   - Institutions: "University", "Institute" → INVALID
   - Title words: "Economics", "Analysis" → INVALID

VALID surnames:
- Normal surnames: "Smith", "McCrary", "Thursby", "Nakamura"
- With prefixes: "de Chaisemartin", "van der Berg", "O'Brien"
- Concatenated first+last: "JonathanRoth" → extract "Roth" only

Examples:
- ["Evidencefromradiologists"] → {"valid_surnames": [], "invalid": ["Evidencefromradiologists"], "reason": "concatenated title words"}
- ["McCrary"] → {"valid_surnames": ["McCrary"], "invalid": [], "reason": "valid surname"}
- ["JonathanRoth"] → {"valid_surnames": ["Roth"], "invalid": [], "reason": "extracted surname from concatenated name"}"""
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

            # FALLBACK: Only if trust_source=True (WebSearch results)
            # For untrusted sources, accept GPT's judgment even if empty
            if not valid and original_valid and trust_source:
                return original_valid, False

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
                    return original_valid if original_valid else surnames, False
            else:
                print(f"GPT surname validation error: {e}")
                return original_valid if original_valid else surnames, False

    return original_valid if original_valid else surnames, False


print("GPT functions defined.")


# --- Year extraction fallback functions ---

def extract_year_from_title(title: str) -> Optional[str]:
    """
    Extract publication year from title string.
    Looks for 4-digit years (1960-2026) at the end or in parentheses.
    """
    if not title:
        return None

    # Pattern 1: Year at the end of title "... 2022" or "... (2022)"
    match = re.search(r'\b(19[6-9]\d|20[0-2]\d)\s*$', title)
    if match:
        return match.group(1)

    # Pattern 2: Year in parentheses "(2022)"
    match = re.search(r'\((\d{4})\)', title)
    if match:
        y = int(match.group(1))
        if 1960 <= y <= 2026:
            return match.group(1)

    # Pattern 3: "Working Paper XXXXX" - NBER papers often have year in title
    match = re.search(r'Working\s+Paper\s+(?:No\.?\s*)?(\d+)', title, re.IGNORECASE)
    if match:
        wp_num = int(match.group(1))
        # NBER WP numbers roughly correspond to years (very approximate)
        year = nber_wp_to_year(wp_num)
        if year:
            return year

    return None


def extract_year_from_filename(filename: str) -> Optional[str]:
    """
    Extract year from original filename.
    Many downloaded PDFs have year in the filename.
    """
    if not filename:
        return None

    # Remove extension
    name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)

    # Pattern 1: Year at the end "Author_2022" or "Author2022"
    match = re.search(r'[_\-]?(19[6-9]\d|20[0-2]\d)(?:[_\-][a-z])?$', name)
    if match:
        return match.group(1)

    # Pattern 2: Year anywhere in filename
    matches = re.findall(r'\b(19[6-9]\d|20[0-2]\d)\b', name)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple years - return the most recent (likely publication year)
        return max(matches)

    return None


def nber_wp_to_year(wp_number: int) -> Optional[str]:
    """
    Estimate publication year from NBER Working Paper number.
    Based on approximate WP number ranges.

    NBER WP numbering:
    - ~1000 in 1980
    - ~5000 in 1995
    - ~10000 in 2003
    - ~15000 in 2009
    - ~20000 in 2014
    - ~25000 in 2018
    - ~30000 in 2022
    """
    if wp_number < 1000:
        return None
    elif wp_number < 2000:
        year = 1980 + (wp_number - 1000) // 200
    elif wp_number < 5000:
        year = 1985 + (wp_number - 2000) // 300
    elif wp_number < 10000:
        year = 1995 + (wp_number - 5000) // 625
    elif wp_number < 15000:
        year = 2003 + (wp_number - 10000) // 830
    elif wp_number < 20000:
        year = 2009 + (wp_number - 15000) // 1000
    elif wp_number < 25000:
        year = 2014 + (wp_number - 20000) // 1250
    elif wp_number < 30000:
        year = 2018 + (wp_number - 25000) // 1250
    elif wp_number < 35000:
        year = 2022 + (wp_number - 30000) // 1500
    else:
        year = 2025

    if 1960 <= year <= 2026:
        return str(year)
    return None


print("Year extraction fallback functions defined.")


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

        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
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
# Cell 5.5: Reset files with bad title extraction (JSTOR boilerplate problem)
# - Identifies files where title extraction captured copyright text instead of real title
# - Resets these entries so they will be reprocessed with improved extraction

RESET_BAD_TITLES = True  # Set to False to skip this step

if RESET_BAD_TITLES:
    print("\n=== Cell 5.5: Resetting files with bad title extraction ===")

    # Load progress to check for bad titles
    progress_check = load_progress()

    # Boilerplate indicators that should NOT appear in titles
    bad_title_indicators = [
        'collaborating with jstor',
        'jstor is a not-for-profit',
        'american economic association is collaborating',
        'your use of the jstor archive',
        'terms and conditions',
        'this content downloaded',
        'all use subject to',
        'accessed:',
        'please contact jstor'
    ]

    # Also identify files that all got the same wrong authors (Bernstein problem)
    author_counts = {}
    for h, entry in progress_check['by_hash'].items():
        fa = tuple(entry.get('final_authors', []))
        fy = entry.get('final_year')
        if fa and fy:
            key = (fa, fy)
            if key not in author_counts:
                author_counts[key] = []
            author_counts[key].append(h)

    # Find author/year combinations with suspiciously many files (>5 is suspicious)
    suspicious_combos = {k: v for k, v in author_counts.items() if len(v) > 5}

    reset_count = 0
    reset_hashes = set()

    # Reset files with bad title indicators
    for file_hash, entry in progress_check['by_hash'].items():
        title = (entry.get('title') or '').lower()
        should_reset = False

        # Check for boilerplate in title
        if any(ind in title for ind in bad_title_indicators):
            should_reset = True

        # Check for suspicious author/year combo
        fa = tuple(entry.get('final_authors', []))
        fy = entry.get('final_year')
        if (fa, fy) in suspicious_combos:
            should_reset = True

        # Check for concatenated title words used as authors
        # e.g., "Evidencefromradiologists", "Evidencefromthecourts"
        embedded_words = ['from', 'with', 'evidence', 'effect', 'model',
                          'market', 'policy', 'average', 'treatment',
                          'selection', 'patent', 'innovation', 'court',
                          'experiment', 'estimat', 'causal', 'empiric']
        for author in entry.get('final_authors', []):
            if author and len(author) > 10:
                author_lower = author.lower()
                for word in embedded_words:
                    if word in author_lower and author_lower.find(word) > 0:
                        should_reset = True
                        break

        if should_reset:
            reset_hashes.add(file_hash)
            reset_count += 1

    if reset_count > 0:
        print(f"Found {reset_count} files with bad title extraction")
        print(f"Suspicious author/year combos (>5 files):")
        for (authors, year), hashes in suspicious_combos.items():
            print(f"  {authors} {year}: {len(hashes)} files")

        # Reset these entries
        for file_hash in reset_hashes:
            if file_hash in progress_check['by_hash']:
                # Keep minimal info, reset everything else for reprocessing
                entry = progress_check['by_hash'][file_hash]
                progress_check['by_hash'][file_hash] = {
                    'filename': entry.get('filename'),
                    'file_hash': file_hash,
                    'reset_reason': 'bad_title_extraction',
                    'original_title': entry.get('title', '')[:50] + '...' if entry.get('title') else None
                }

        save_progress(progress_check)
        print(f"Reset {reset_count} entries for reprocessing")
    else:
        print("No files with bad title extraction found")


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

# Detect files returned from failure/ or re-search/ and clear their cache for re-processing
# This allows users to manually move files back to ARTICLES_DIR for another attempt
returned_count = 0
returned_from = {'failure': 0, 're-search': 0}
for file_hash in list(progress['by_hash'].keys()):
    entry = progress['by_hash'][file_hash]
    moved_to = entry.get('moved_to')
    if moved_to in ('failure', 're-search'):
        if file_hash in current_hashes:
            # File has returned to ARTICLES_DIR - clear cache for re-processing
            old_filename = entry.get('filename', 'unknown')
            print(f"  Re-processing: {old_filename} (returned from {moved_to}/)")
            del progress['by_hash'][file_hash]
            if file_hash in progress.get('hash_to_filename', {}):
                del progress['hash_to_filename'][file_hash]
            returned_count += 1
            returned_from[moved_to] += 1

if returned_count > 0:
    save_progress(progress)
    print(f"Cleared cache for {returned_count} returned files:")
    print(f"  - From failure/: {returned_from['failure']}")
    print(f"  - From re-search/: {returned_from['re-search']}")

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

# Process WebSearch with filename parsing and title normalization
search_count = 0
filename_verified_count = 0
title_normalized_count = 0

for i, file_hash in enumerate(to_search):
    entry = progress['by_hash'][file_hash]
    filename = entry['filename']

    if (i + 1) % 10 == 0:
        print(f"Searching {i+1}/{len(to_search)}... ({filename[:30]})")

    # Step 1: Check if filename already follows Author_Year.pdf format
    fn_authors, fn_year, is_valid_format = parse_existing_filename(filename)

    if is_valid_format and fn_authors:
        # Filename looks valid - try to verify with WebSearch using first author
        # Search with author name + year to confirm paper exists
        verify_query = f"{fn_authors[0]} {fn_year}"
        ws_authors, ws_year, ws_source = websearch_authors(entry.get('title') or verify_query)

        if ws_authors:
            # WebSearch found results - check if they match filename
            ws_surnames = [a.lower() for a in ws_authors]
            fn_surnames = [a.lower() for a in fn_authors]

            # Check if at least first author matches
            if fn_surnames[0] in ws_surnames or ws_surnames[0] in [a.lower() for a in fn_authors]:
                # Verified - use WebSearch results but mark as filename-verified
                entry['websearch_authors'] = ws_authors
                entry['websearch_year'] = ws_year
                entry['websearch_source'] = ws_source
                entry['websearch_status'] = 'done'
                entry['filename_verified'] = True
                filename_verified_count += 1
                entry['websearch_at'] = datetime.now().isoformat()
                search_count += 1

                if search_count % SAVE_INTERVAL == 0:
                    save_progress(progress)
                    print(f"Progress saved at {search_count}")
                continue

            # Authors don't match - but WebSearch found valid authors
            # Use WebSearch results (more reliable than filename which might be wrong)
            # e.g., "Edicine_2009.pdf" where "Edicine" is not an author but WebSearch finds "Austin"
            entry['websearch_authors'] = ws_authors
            entry['websearch_year'] = ws_year
            entry['websearch_source'] = ws_source
            entry['websearch_status'] = 'done'
            entry['filename_mismatch'] = True  # Flag that filename didn't match
            entry['websearch_at'] = datetime.now().isoformat()
            search_count += 1

            if search_count % SAVE_INTERVAL == 0:
                save_progress(progress)
                print(f"Progress saved at {search_count}")
            continue

        # WebSearch didn't find any authors - trust the filename format
        # (human-assigned names are reliable when no other source available)
        entry['websearch_authors'] = []
        entry['websearch_year'] = None
        entry['websearch_source'] = None
        entry['websearch_status'] = 'not_found'
        entry['filename_trusted'] = True  # Mark as trusted from filename
        entry['filename_authors'] = fn_authors
        entry['filename_year'] = fn_year
        entry['websearch_at'] = datetime.now().isoformat()
        filename_verified_count += 1
        search_count += 1

        if search_count % SAVE_INTERVAL == 0:
            save_progress(progress)
            print(f"Progress saved at {search_count}")
        continue

    # Step 2: Check if title needs normalization (garbled/concatenated text)
    title = entry.get('title', '')
    search_title = title

    # Clean special characters that may interfere with WebSearch
    if title:
        search_title = re.sub(r'[†‡§¶*∗]+', '', title).strip()

    if title:
        # Detect garbled title: has very long "words" (concatenated text)
        words = title.split()
        has_long_words = any(len(w) > 30 for w in words) if words else False
        few_words = len(words) < 3

        if has_long_words or few_words:
            # Try to normalize with GPT
            normalized = normalize_title_with_gpt(title)
            if normalized and normalized != title:
                search_title = normalized
                entry['title_normalized'] = normalized
                title_normalized_count += 1

    # Step 3: Perform WebSearch with (possibly normalized) title
    ws_authors, ws_year, ws_source = websearch_authors(search_title)

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
print(f"  - Filename verified (Author_Year.pdf format): {filename_verified_count}")
print(f"  - Title normalized by GPT: {title_normalized_count}")
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
    # Priority: WebSearch > Filename (trusted) > Metadata
    if ws_valid:
        item['final_authors'] = ws_valid
        item['author_source'] = 'websearch'
    elif item.get('filename_trusted') and item.get('filename_authors'):
        # Filename was in Author_Year format and trusted
        item['final_authors'] = item['filename_authors']
        item['author_source'] = 'filename'
    elif meta_valid:
        item['final_authors'] = meta_valid
        item['author_source'] = 'metadata'
    else:
        item['final_authors'] = []
        item['author_source'] = 'none'
        item['status'] = 'alert'  # Need PDF text search
    
    # Determine final year
    # Priority: Metadata > WebSearch > Filename
    # Note: Title extraction removed (too error-prone)
    if meta_year:
        item['final_year'] = meta_year
        item['year_source'] = 'metadata'
    elif ws_year:
        item['final_year'] = ws_year
        item['year_source'] = 'websearch'
    else:
        # Fallback: Extract from original filename
        filename_year = extract_year_from_filename(filename)
        if filename_year:
            item['final_year'] = filename_year
            item['year_source'] = 'filename'
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
        # Have authors but no year - use n.d. (no date) and mark as alert
        item['status'] = 'alert'
        item['alert_reason'] = 'no_year_use_nd'
        item['final_year'] = 'n.d.'
        item['year_source'] = 'fallback_nd'

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
# Cell 9.6: Apply existing OCR+GPT results that weren't applied
# - Finds files where ocr_gpt_authors exists but final_authors is empty
# - Applies the OCR+GPT results and changes status to alert

print("\n=== Cell 9.6: Apply existing OCR+GPT results ===")

ocr_results_applied = 0
for item in pdf_data:
    ocr_authors = item.get('ocr_gpt_authors', [])
    final_authors = item.get('final_authors', [])
    status = item.get('status')

    # If we have OCR+GPT authors but no final_authors, apply them
    if ocr_authors and not final_authors and status == 'fail':
        # Validate the OCR authors
        valid_authors = [a for a in ocr_authors if is_valid_surname(a)]
        if valid_authors:
            item['final_authors'] = valid_authors
            item['author_source'] = 'ocr_gpt_fallback_recovered'

            # Also apply year if available
            ocr_year = item.get('ocr_gpt_year')
            if ocr_year and not item.get('final_year'):
                item['final_year'] = ocr_year
                item['year_source'] = 'ocr_gpt_fallback'

            # Change status to alert
            if item.get('final_year'):
                item['status'] = 'alert'
                item['alert_reason'] = 'ocr_gpt_recovered'
                if 'fail_reason' in item:
                    del item['fail_reason']
                ocr_results_applied += 1
                print(f"  Applied: {item['filename']} → {valid_authors}")

print(f"\nApplied OCR+GPT results to {ocr_results_applied} files")

# Update progress
for item in pdf_data:
    file_hash = item.get('file_hash', '')
    if file_hash and file_hash in progress['by_hash']:
        progress['by_hash'][file_hash].update({
            'status': item['status'],
            'final_authors': item.get('final_authors'),
            'final_year': item.get('final_year'),
            'author_source': item.get('author_source'),
            'alert_reason': item.get('alert_reason'),
            'fail_reason': item.get('fail_reason')
        })
save_progress(progress)

# Update counts
success_count = sum(1 for x in pdf_data if x['status'] == 'success')
fail_count = sum(1 for x in pdf_data if x['status'] == 'fail')
alert_count = sum(1 for x in pdf_data if x['status'] == 'alert')
print(f"\nStatus after applying OCR+GPT results:")
print(f"  Success: {success_count}")
print(f"  Alert: {alert_count}")
print(f"  Fail: {fail_count}")


# %%
# Cell 10: Generate new filenames

def generate_filename(authors: List[str], year: str, existing: set) -> Optional[str]:
    """Generate filename from authors and year.

    Naming convention:
    - 1 author: Author_Year.pdf
    - 2 authors: Author_Author_Year.pdf
    - 3 authors: Author_Author_Author_Year.pdf
    - 4+ authors: Author_et_al_Year.pdf
    """
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
    elif len(authors) == 3:
        author_part = f"{authors[0]}_{authors[1]}_{authors[2]}"
    else:
        # 4 or more authors: use et_al
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


# Generate new names with GPT validation as final check
existing = set()
GPT_FINAL_VALIDATION = True  # Final GPT check before renaming

print("Generating filenames...")
if GPT_FINAL_VALIDATION:
    print("  GPT final validation enabled")

gpt_rejected_count = 0

for i, item in enumerate(pdf_data):
    if (i + 1) % 200 == 0:
        print(f"  Processing {i+1}/{len(pdf_data)}...")

    if item['status'] == 'success':
        authors = item.get('final_authors', [])
        title = item.get('title', '')

        # Final GPT validation for non-trusted sources
        # WebSearch (CrossRef, Semantic Scholar) and Filename (human-assigned) are trusted
        author_source = item.get('author_source', '')
        if GPT_FINAL_VALIDATION and authors and author_source not in ('websearch', 'filename'):
            validated, _ = validate_surnames_with_gpt(authors, title, trust_source=False)
            if not validated:
                # GPT rejected all authors - move to fail
                item['status'] = 'fail'
                item['fail_reason'] = 'gpt_final_validation_failed'
                item['gpt_rejected_authors'] = authors
                item['new_filename'] = None
                item['move_to_failure'] = True
                gpt_rejected_count += 1
                continue
            elif validated != authors:
                # GPT corrected authors
                item['final_authors'] = validated
                item['gpt_final_corrected'] = True
                authors = validated

        new_name = generate_filename(authors, item['final_year'], existing)
        if new_name:
            item['new_filename'] = new_name
            existing.add(new_name.lower())
        else:
            item['new_filename'] = None
            item['status'] = 'fail'
            item['fail_reason'] = 'filename_generation_failed'

    elif item['status'] == 'alert':
        authors = item.get('final_authors', [])
        title = item.get('title', '')

        # GPT validation for alert files too
        if GPT_FINAL_VALIDATION and authors:
            validated, _ = validate_surnames_with_gpt(authors, title, trust_source=False)
            if validated:
                item['final_authors'] = validated
                authors = validated

        # Alert: generate filename but treat as success (no move to alert/)
        # Alert list will be saved separately for reference
        new_name = generate_filename(authors, item['final_year'], existing)
        if new_name:
            item['new_filename'] = new_name
            existing.add(new_name.lower())
            item['is_alert'] = True  # Mark for alert list recording
        else:
            item['new_filename'] = None
            item['status'] = 'fail'
            item['fail_reason'] = 'filename_generation_failed'

    elif item['status'] == 'fail':
        # Fail: no filename, move to failure/ directory
        item['new_filename'] = None
        item['move_to_failure'] = True
    # Japanese files keep their original name and move to japanese/

if GPT_FINAL_VALIDATION:
    print(f"  GPT rejected {gpt_rejected_count} files with invalid authors")

success_count = sum(1 for x in pdf_data if x['status'] == 'success' and x.get('new_filename'))
alert_count = sum(1 for x in pdf_data if x.get('is_alert') and x.get('new_filename'))
failure_count = sum(1 for x in pdf_data if x.get('move_to_failure'))
japanese_count = sum(1 for x in pdf_data if x['status'] == 'japanese')

print(f"Filename generation complete.")
print(f"  Files to rename (success): {success_count}")
print(f"  Files to rename (alert, for reference): {alert_count}")
print(f"  Files to move to failure/: {failure_count}")
print(f"  Files to move to japanese/: {japanese_count}")

# Update progress with Cell 10 changes (GPT validation results)
for item in pdf_data:
    file_hash = item.get('file_hash', '')
    if file_hash and file_hash in progress['by_hash']:
        progress['by_hash'][file_hash].update({
            'status': item['status'],
            'final_authors': item.get('final_authors'),
            'fail_reason': item.get('fail_reason'),
            'gpt_rejected_authors': item.get('gpt_rejected_authors'),
            'gpt_final_corrected': item.get('gpt_final_corrected'),
        })
save_progress(progress)
print("Progress saved after filename generation.")

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
    failure_dir = os.path.join(ARTICLES_DIR, 'failure')
    for folder in [japanese_dir, research_dir, failure_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    renamed = []
    renamed_alert = []  # Track alert files separately (for reference only)
    moved_japanese = []
    moved_research = []
    moved_failure = []
    errors = []
    moved_hash_duplicates = []  # Files with same hash as another file

    # --- Pre-process: Handle files not in pdf_data (hash collision orphans) ---
    # These are files that weren't processed because another file with same hash already existed
    pdf_data_filenames = {item['filename'] for item in pdf_data}
    all_article_pdfs = [f for f in os.listdir(ARTICLES_DIR)
                        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(ARTICLES_DIR, f))]
    orphan_files = [f for f in all_article_pdfs if f not in pdf_data_filenames]

    if orphan_files:
        print(f"Found {len(orphan_files)} files not in progress data (hash collision orphans)")

        # Build hash -> filename mapping from pdf_data
        hash_to_processed = {}
        for item in pdf_data:
            file_hash = item.get('file_hash')
            if file_hash:
                hash_to_processed[file_hash] = item

        # Create duplicate folder
        dup_dir = os.path.join(ARTICLES_DIR, 'duplicate')
        if not os.path.exists(dup_dir):
            os.makedirs(dup_dir)

        for orphan in orphan_files:
            orphan_path = os.path.join(ARTICLES_DIR, orphan)
            if not os.path.exists(orphan_path):
                continue

            orphan_hash = calculate_file_hash(orphan_path)

            # Check if this hash was processed under a different filename
            if orphan_hash in hash_to_processed:
                processed_item = hash_to_processed[orphan_hash]
                processed_filename = processed_item.get('new_filename') or processed_item['filename']

                # This file is a duplicate - move to duplicate folder
                dup_path = os.path.join(dup_dir, orphan)
                if os.path.exists(dup_path):
                    base, ext = os.path.splitext(orphan)
                    dup_path = os.path.join(dup_dir, f"{base}_{datetime.now().strftime('%H%M%S')}{ext}")

                try:
                    os.rename(orphan_path, dup_path)
                    moved_hash_duplicates.append({
                        'old': orphan,
                        'new': f'duplicate/{os.path.basename(dup_path)}',
                        'duplicate_of': processed_filename
                    })
                    print(f"  Moved hash duplicate: {orphan} -> duplicate/ (same as {processed_filename})")
                except Exception as e:
                    errors.append({'old': orphan, 'new': f'duplicate/{orphan}', 'reason': str(e)})

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

        # Handle OCR-only papers - move to re-search folder for manual verification
        # Check OCR BEFORE failure so OCR files go to re-search even if they failed
        title_method = item.get('title_method') or ''
        if title_method.startswith('ocr'):
            new_name = item.get('new_filename') or old_name  # Use original name if no new name
            new_path = os.path.join(research_dir, new_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_research.append({'old': old_name, 'new': f're-search/{new_name}'})
                    # Record moved_to flag for re-processing detection
                    file_hash = item.get('file_hash')
                    if file_hash and file_hash in progress['by_hash']:
                        progress['by_hash'][file_hash]['moved_to'] = 're-search'
                        progress['by_hash'][file_hash]['moved_at'] = datetime.now().isoformat()
                else:
                    errors.append({'old': old_name, 'new': f're-search/{new_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f're-search/{new_name}', 'reason': str(e)})
            continue

        # Handle fail papers - move to failure folder (no rename)
        if item.get('move_to_failure'):
            new_path = os.path.join(failure_dir, old_name)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    moved_failure.append({'old': old_name, 'new': f'failure/{old_name}',
                                         'fail_reason': item.get('fail_reason')})
                    # Record moved_to flag for re-processing detection
                    file_hash = item.get('file_hash')
                    if file_hash and file_hash in progress['by_hash']:
                        progress['by_hash'][file_hash]['moved_to'] = 'failure'
                        progress['by_hash'][file_hash]['moved_at'] = datetime.now().isoformat()
                else:
                    errors.append({'old': old_name, 'new': f'failure/{old_name}', 'reason': 'path conflict'})
            except Exception as e:
                errors.append({'old': old_name, 'new': f'failure/{old_name}', 'reason': str(e)})
            continue

        # Handle regular renaming (includes alert files - they are renamed but tracked separately)
        new_name = item.get('new_filename')

        if not new_name or old_name == new_name:
            continue

        new_path = os.path.join(ARTICLES_DIR, new_name)

        try:
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                rename_entry = {'old': old_name, 'new': new_name}
                renamed.append(rename_entry)
                # Track alert files separately for reference
                if item.get('is_alert'):
                    renamed_alert.append({
                        'old': old_name,
                        'new': new_name,
                        'alert_reason': item.get('alert_reason'),
                        'authors': item.get('final_authors'),
                        'author_source': item.get('author_source')
                    })
            else:
                # Path conflict: check if same content (duplicate) or different content (need suffix)
                old_hash = calculate_file_hash(old_path)
                existing_hash = calculate_file_hash(new_path)

                if old_hash == existing_hash:
                    # Same content - move to duplicate folder
                    dup_dir = os.path.join(ARTICLES_DIR, 'duplicate')
                    if not os.path.exists(dup_dir):
                        os.makedirs(dup_dir)
                    dup_path = os.path.join(dup_dir, old_name)
                    if os.path.exists(dup_path):
                        # Add timestamp if duplicate name exists
                        base, ext = os.path.splitext(old_name)
                        dup_path = os.path.join(dup_dir, f"{base}_{datetime.now().strftime('%H%M%S')}{ext}")
                    os.rename(old_path, dup_path)
                    renamed.append({'old': old_name, 'new': f'duplicate/{os.path.basename(dup_path)}',
                                   'note': 'same content as existing file'})
                else:
                    # Different content - add disambiguation suffix
                    base_name, ext = os.path.splitext(new_name)
                    suffix_char = 'a'
                    while True:
                        suffixed_name = f"{base_name}_{suffix_char}{ext}"
                        suffixed_path = os.path.join(ARTICLES_DIR, suffixed_name)
                        if not os.path.exists(suffixed_path):
                            os.rename(old_path, suffixed_path)
                            renamed.append({'old': old_name, 'new': suffixed_name,
                                           'note': 'added suffix due to name conflict'})
                            break
                        suffix_char = chr(ord(suffix_char) + 1)
                        if suffix_char > 'z':
                            errors.append({'old': old_name, 'new': new_name,
                                          'reason': 'path conflict - exhausted suffixes'})
                            break
        except Exception as e:
            errors.append({'old': old_name, 'new': new_name, 'reason': str(e)})

    print(f"\nRename complete:")
    print(f"  Renamed: {len(renamed)} (including {len(renamed_alert)} alert files)")
    print(f"  Moved to japanese/: {len(moved_japanese)}")
    print(f"  Moved to failure/: {len(moved_failure)}")
    print(f"  Moved to re-search/: {len(moved_research)}")
    print(f"  Moved hash duplicates: {len(moved_hash_duplicates)}")
    print(f"  Errors: {len(errors)}")

    # Show alert details (for reference)
    if renamed_alert:
        print(f"\n--- Alert files (renamed, for reference) ---")
        for item in renamed_alert[:20]:
            print(f"  {item['old']} -> {item['new']}")
            print(f"    Reason: {item.get('alert_reason')}, Authors: {item.get('authors')}")

    # Save results
    with open(os.path.join(DATA_DIR, 'rename_results_final.json'), 'w') as f:
        json.dump({
            'renamed': renamed,
            'renamed_alert': renamed_alert,
            'moved_japanese': moved_japanese,
            'moved_failure': moved_failure,
            'moved_research': moved_research,
            'moved_hash_duplicates': moved_hash_duplicates,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    # Save alert list separately for reference
    if renamed_alert:
        with open(os.path.join(DATA_DIR, 'alert_files_list.json'), 'w') as f:
            json.dump({
                'description': 'Files renamed with low confidence (OCR+GPT fallback). Review if issues found.',
                'count': len(renamed_alert),
                'files': renamed_alert,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        print(f"\nAlert list saved to {os.path.join(DATA_DIR, 'alert_files_list.json')}")

    # Save progress with moved_to flags for re-processing detection
    save_progress(progress)
    print("Progress cache saved with moved_to flags.")
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
    ocr_files = [x for x in pdf_data if (x.get('title_method') or '').startswith('ocr')]
    if ocr_files:
        print(f"\nOCR-only files to be moved to re-search/ ({len(ocr_files)}):")
        for item in ocr_files[:10]:
            new_name = item.get('new_filename') or item['filename']
            print(f"  {item['filename']} -> re-search/{new_name}")

# %%
# Cell 13: Save final results

# Save all data
with open(os.path.join(DATA_DIR, 'pdf_rename_data.json'), 'w') as f:
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

with open(os.path.join(DATA_DIR, 'pdf_rename_summary.json'), 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Results saved:")
print(f"  - pdf_rename_data.json (full data)")
print(f"  - pdf_rename_summary.json (summary)")
print()
print(json.dumps(summary, indent=2))

# %%
# Cell 13.5: Generate detailed failure report
# - Documents why each file failed
# - Records which processing steps were attempted
# - Saves to failure_report.json for debugging

print("\n=== Generating Failure Report ===")

fail_files = [x for x in pdf_data if x['status'] == 'fail']
print(f"Total failure files: {len(fail_files)}")

failure_report = {
    'generated_at': datetime.now().isoformat(),
    'total_failures': len(fail_files),
    'failure_summary': {},
    'files': []
}

# Count failure reasons
for item in fail_files:
    reason = item.get('fail_reason', 'unknown')
    failure_report['failure_summary'][reason] = failure_report['failure_summary'].get(reason, 0) + 1

# Generate detailed report for each failure
for item in fail_files:
    # Determine processing progress
    steps_completed = []
    steps_failed = []

    # Step 1: Title extraction
    if item.get('title'):
        steps_completed.append('title_extraction')
    else:
        steps_failed.append('title_extraction')

    # Step 2: WebSearch
    if item.get('websearch_status') == 'done':
        if item.get('websearch_authors'):
            steps_completed.append('websearch')
        else:
            steps_failed.append('websearch (no results)')
    elif item.get('websearch_status') == 'not_found':
        steps_failed.append('websearch (not found)')
    elif item.get('title'):
        steps_failed.append('websearch (not attempted or failed)')

    # Step 3: PDF text search
    if item.get('author_source', '').startswith('pdf_'):
        steps_completed.append('pdf_text_search')
    elif item.get('pdf_gpt_validation'):
        steps_completed.append('pdf_text_search (attempted)')

    # Step 4: OCR+GPT fallback
    if item.get('ocr_gpt_attempted'):
        if item.get('ocr_gpt_authors'):
            steps_completed.append('ocr_gpt_fallback')
        else:
            steps_failed.append('ocr_gpt_fallback (no results)')

    # Step 5: GPT final validation
    if item.get('gpt_rejected_authors'):
        steps_failed.append('gpt_final_validation (rejected)')

    # Build detailed entry
    entry = {
        'filename': item.get('filename'),
        'file_hash': item.get('file_hash'),
        'fail_reason': item.get('fail_reason', 'unknown'),
        'processing_progress': {
            'steps_completed': steps_completed,
            'steps_failed': steps_failed,
            'furthest_step': steps_completed[-1] if steps_completed else 'none'
        },
        'extraction_details': {
            'title': item.get('title'),
            'title_method': item.get('title_method'),
            'title_error': item.get('title_error'),
        },
        'author_search_details': {
            'websearch_status': item.get('websearch_status'),
            'websearch_source': item.get('websearch_source'),
            'websearch_authors': item.get('websearch_authors'),
            'meta_authors': item.get('meta_authors'),
            'ocr_gpt_authors': item.get('ocr_gpt_authors'),
            'gpt_rejected_authors': item.get('gpt_rejected_authors'),
        },
        'year_details': {
            'meta_year': item.get('meta_year'),
            'websearch_year': item.get('websearch_year'),
            'final_year': item.get('final_year'),
        },
        'timestamps': {
            'extracted_at': item.get('extracted_at'),
            'websearch_at': item.get('websearch_at'),
        }
    }

    failure_report['files'].append(entry)

# Save failure report
failure_report_path = os.path.join(DATA_DIR, 'failure_report.json')
with open(failure_report_path, 'w') as f:
    json.dump(failure_report, f, ensure_ascii=False, indent=2)

print(f"\nFailure report saved to: {failure_report_path}")
print(f"\nFailure reasons breakdown:")
for reason, count in sorted(failure_report['failure_summary'].items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")

# Show sample of detailed failures
print(f"\nSample failure details (first 5):")
for entry in failure_report['files'][:5]:
    print(f"\n  {entry['filename']}")
    print(f"    Reason: {entry['fail_reason']}")
    print(f"    Steps completed: {entry['processing_progress']['steps_completed']}")
    print(f"    Steps failed: {entry['processing_progress']['steps_failed']}")
    if entry['extraction_details']['title']:
        print(f"    Title: {entry['extraction_details']['title'][:50]}...")
    else:
        print(f"    Title: (none) - error: {entry['extraction_details']['title_error']}")


# %%
# Cell 14: Detect and move duplicate files (Enhanced with content-based detection)
# - Finds files like Name_Year.pdf, Name_Year_a.pdf, Name_Year_b.pdf
# - Uses both MD5 hash AND text similarity (80% threshold, chars 500-2500)
# - Keeps the file with more extractable text
# - Detects Working Papers and renames with _wp suffix
# - Re-evaluates existing files in duplicate/ folder

DETECT_DUPLICATES = True  # Set to False to skip duplicate detection
SIMILARITY_THRESHOLD = 0.80  # 80% text similarity threshold
TEXT_START = 500  # Start character position for comparison
TEXT_END = 2500   # End character position for comparison

# Working Paper keywords to detect
WP_KEYWORDS = [
    'working paper',
    'discussion paper',
    'nber working',
    'nber wp',
    'cepr discussion',
    'iza discussion',
    'ssrn',
    'preliminary draft',
    'do not cite',
    'work in progress'
]


def extract_pdf_text_for_comparison(pdf_path: str, max_chars: int = 3000) -> str:
    """Extract text from PDF for comparison (first max_chars characters)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
                if len(text) >= max_chars:
                    break
            return text[:max_chars]
    except Exception as e:
        return ""


def calculate_text_similarity(text1: str, text2: str, start: int = 500, end: int = 2500) -> float:
    """
    Calculate similarity between two texts using the specified character range.
    Returns similarity ratio between 0 and 1.
    """
    # Extract the comparison range
    segment1 = text1[start:end].lower().strip()
    segment2 = text2[start:end].lower().strip()

    # If either segment is too short, return 0
    if len(segment1) < 100 or len(segment2) < 100:
        return 0.0

    # Use difflib for sequence matching
    from difflib import SequenceMatcher
    return SequenceMatcher(None, segment1, segment2).ratio()


def is_working_paper(text: str, max_check_chars: int = 3000) -> bool:
    """Check if PDF text contains Working Paper indicators."""
    check_text = text[:max_check_chars].lower()
    return any(kw in check_text for kw in WP_KEYWORDS)


def count_extractable_text(pdf_path: str) -> int:
    """Count total extractable text length from PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = 0
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                total += len(page_text)
            return total
    except Exception as e:
        return 0


if DETECT_DUPLICATES:
    print("=== Detecting duplicate files (Enhanced) ===")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD*100:.0f}%")
    print(f"  Text comparison range: chars {TEXT_START}-{TEXT_END}")

    # Create duplicate folder
    duplicate_dir = os.path.join(ARTICLES_DIR, 'duplicate')
    if not os.path.exists(duplicate_dir):
        os.makedirs(duplicate_dir)
        print(f"Created directory: {duplicate_dir}")

    # --- Step 1: Re-evaluate existing duplicate/ files ---
    print("\n--- Re-evaluating existing duplicate/ files ---")
    existing_duplicates = [f for f in os.listdir(duplicate_dir)
                          if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(duplicate_dir, f))]
    print(f"  Found {len(existing_duplicates)} files in duplicate/")

    restored_files = []
    for dup_file in existing_duplicates:
        # Extract base name to find potential match in ARTICLES_DIR
        match = re.match(r'^(.+?)(_[a-z])?(_wp)?\.pdf$', dup_file, re.IGNORECASE)
        if not match:
            continue

        base_name = match.group(1)
        dup_path = os.path.join(duplicate_dir, dup_file)

        # Find matching files in ARTICLES_DIR
        potential_matches = [f for f in os.listdir(ARTICLES_DIR)
                           if f.lower().endswith('.pdf')
                           and f.startswith(base_name)
                           and os.path.isfile(os.path.join(ARTICLES_DIR, f))]

        if not potential_matches:
            # No match found, restore the file
            restore_path = os.path.join(ARTICLES_DIR, dup_file.replace('_wp.pdf', '.pdf'))
            if not os.path.exists(restore_path):
                os.rename(dup_path, restore_path)
                restored_files.append(dup_file)
                print(f"  Restored: {dup_file} (no matching base file)")

    if restored_files:
        print(f"  Restored {len(restored_files)} files to ARTICLES_DIR")

    # --- Step 2: Get all PDF files in ARTICLES_DIR ---
    pdf_files = [f for f in os.listdir(ARTICLES_DIR)
                 if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(ARTICLES_DIR, f))]

    # Group files by base name (without _a, _b, etc. suffix)
    file_groups = {}  # base_name -> list of (filename, suffix_order)

    for filename in pdf_files:
        # Check if filename has _a, _b, etc. suffix before .pdf
        match = re.match(r'^(.+?)(_[a-z])?\.pdf$', filename, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            suffix = match.group(2)  # None or '_a', '_b', etc.

            if base_name not in file_groups:
                file_groups[base_name] = []

            # Assign order: no suffix = 0, _a = 1, _b = 2, etc.
            if suffix is None:
                order = 0
            else:
                order = ord(suffix[-1].lower()) - ord('a') + 1

            file_groups[base_name].append((filename, order))

    # Find groups with potential duplicates (more than one file)
    duplicate_candidates = {k: v for k, v in file_groups.items() if len(v) > 1}
    print(f"\n--- Processing {len(duplicate_candidates)} file groups with potential duplicates ---")

    # --- Step 3: Check for duplicates using MD5 hash AND text similarity ---
    moved_duplicates = []
    hash_matches = 0
    similarity_matches = 0

    for base_name, files in duplicate_candidates.items():
        # Sort by order (base file first, then _a, _b, etc.)
        files.sort(key=lambda x: x[1])

        # Calculate hashes and extract text for all files
        file_data = {}  # filename -> {'hash': ..., 'text': ..., 'text_len': ...}
        for filename, _ in files:
            path = os.path.join(ARTICLES_DIR, filename)
            file_hash = calculate_file_hash(path)
            text = extract_pdf_text_for_comparison(path)
            text_len = count_extractable_text(path)
            file_data[filename] = {
                'hash': file_hash,
                'text': text,
                'text_len': text_len,
                'is_wp': is_working_paper(text)
            }

        # Find the best file to keep (most extractable text)
        best_file = max(files, key=lambda x: file_data[x[0]]['text_len'])[0]
        best_data = file_data[best_file]

        # Compare all other files against the best file
        for filename, order in files:
            if filename == best_file:
                continue

            current_data = file_data[filename]
            is_duplicate = False
            match_type = None

            # Check 1: MD5 hash match (exact duplicate)
            if current_data['hash'] == best_data['hash']:
                is_duplicate = True
                match_type = 'hash'
                hash_matches += 1

            # Check 2: Text similarity match (content-based duplicate)
            elif best_data['text'] and current_data['text']:
                similarity = calculate_text_similarity(
                    best_data['text'], current_data['text'],
                    TEXT_START, TEXT_END
                )
                if similarity >= SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    match_type = f'similarity:{similarity:.1%}'
                    similarity_matches += 1

            if is_duplicate:
                # Determine if this is a Working Paper
                is_wp = current_data['is_wp']

                # Determine new filename
                if is_wp:
                    # Rename with _wp suffix
                    new_filename = re.sub(r'(_[a-z])?\.pdf$', '_wp.pdf', filename, flags=re.IGNORECASE)
                else:
                    new_filename = filename

                old_path = os.path.join(ARTICLES_DIR, filename)
                new_path = os.path.join(duplicate_dir, new_filename)

                try:
                    # Handle existing file in duplicate/
                    if os.path.exists(new_path):
                        # Add timestamp to avoid overwrite
                        base, ext = os.path.splitext(new_filename)
                        new_filename = f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
                        new_path = os.path.join(duplicate_dir, new_filename)

                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)
                        moved_duplicates.append({
                            'original_filename': filename,
                            'moved_filename': new_filename,
                            'base_file': best_file,
                            'match_type': match_type,
                            'is_wp': is_wp,
                            'text_len': current_data['text_len'],
                            'best_text_len': best_data['text_len']
                        })
                        wp_tag = " [WP]" if is_wp else ""
                        print(f"  Moved: {filename} -> {new_filename}{wp_tag} ({match_type}, kept {best_file})")
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")

    print(f"\n=== Duplicate detection complete ===")
    print(f"  Hash matches: {hash_matches}")
    print(f"  Similarity matches: {similarity_matches}")
    print(f"  Total moved to duplicate/: {len(moved_duplicates)}")

    # Save duplicate info to JSON
    duplicate_info = {
        'moved_duplicates': moved_duplicates,
        'restored_files': restored_files,
        'settings': {
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'text_range': [TEXT_START, TEXT_END],
            'wp_keywords': WP_KEYWORDS
        },
        'statistics': {
            'hash_matches': hash_matches,
            'similarity_matches': similarity_matches,
            'total_moved': len(moved_duplicates),
            'restored': len(restored_files)
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(DATA_DIR, 'duplicate_files.json'), 'w') as f:
        json.dump(duplicate_info, f, ensure_ascii=False, indent=2)
    print(f"  Saved info to duplicate_files.json")
else:
    print("Duplicate detection skipped (DETECT_DUPLICATES = False)")
