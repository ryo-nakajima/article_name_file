# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF rename tool for academic papers. Extracts titles from PDFs, searches for author information via web APIs, and renames files to `Author_Year.pdf` format.

## Commands

```bash
# Sync .py to .ipynb (after editing the Python file)
cd code && jupytext --to notebook pdf_rename_websearch.py -o pdf_rename_websearch.ipynb

# Run the workflow
# Execute cells sequentially in pdf_rename_websearch.ipynb

# Clear cache and reprocess all files
rm code/data/*.json
```

## Architecture

### Single-file workflow: `code/pdf_rename_websearch.py`

Jupytext percent-format Python file that syncs to `pdf_rename_websearch.ipynb`. Execute cells in order:

| Cell | Purpose |
|------|---------|
| 1-2 | Setup, imports, NON_SURNAMES list |
| 3-3b | Helper functions, GPT-based validation |
| 4 | PDF text search functions |
| 5 | WebSearch (Semantic Scholar + CrossRef APIs) |
| 5.5 | JSTOR boilerplate title reset |
| 6 | Step 1: Extract titles/metadata (incremental) |
| 7 | Step 2: WebSearch for authors (incremental) |
| 8 | Step 3: Compare results, determine final authors/year |
| 9-9.6 | Step 4: PDF text search fallback, OCR+GPT recovery |
| 10 | Generate new filenames |
| 11 | Preview changes |
| 12 | Execute renaming (handles path conflicts, hash duplicates) |
| 13 | Save final results |
| 14 | Detect and move duplicate files |

### Processing Flow

1. **Title extraction**: pdfplumber text → font-based detection → GPT validation → OCR fallback
2. **Author lookup**: Semantic Scholar API → CrossRef API fallback
3. **Validation**: GPT validates author names against NON_SURNAMES list
4. **File organization**: Renamed files stay in `articles/`, others go to `japanese/`, `failure/`, `re-search/`, or `duplicate/`

### Key Data Structures

- **Progress cache**: `code/data/pdf_processing_progress.json` - keyed by MD5 hash
- **Status values**: `success`, `fail`, `alert`, `japanese`
- **Title methods**: `text`, `text_gpt`, `text_jstor`, `ocr`, `ocr_gpt`

### Important Behaviors

- Incremental processing: skips already-processed files based on hash
- Path conflict resolution: same content → `duplicate/`, different content → add `_a`, `_b` suffix
- Hash collision handling: files not in progress but matching existing hash → auto-move to `duplicate/`
- JSTOR boilerplate detection: rejects copyright text as titles, parses cover page structure

## File Organization

```
/Users/ryo2/Dropbox/Articles/     # PDF files (working directory)
├── code/
│   ├── pdf_rename_websearch.py   # Main workflow (edit this)
│   ├── pdf_rename_websearch.ipynb # Generated notebook (run this)
│   └── data/                      # Cache JSON files (gitignored)
├── japanese/                      # Japanese papers
├── failure/                       # Processing failures
├── re-search/                     # OCR files needing review
└── duplicate/                     # Duplicate content files
```
