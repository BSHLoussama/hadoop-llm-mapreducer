import requests, json, time, datetime, random, argparse, os, textwrap, pathlib, sys
from bs4 import BeautifulSoup

# ─── Parser choice ────────────────────────────────────────────────────────────
try:
    import lxml   # noqa
    PARSER = "xml"
except ImportError:
    PARSER = "html.parser"

API_KEY   = os.getenv("NCBI_API_KEY")
ESEARCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ─── configurable search space ────────────────────────────────────────────────
YEARS_SOTA   = 5
YEARS_LEGACY = 25
BATCH        = 200                             # ≤ 200 ids / EFetch request
QUERIES = [
    "generalized anxiety disorder",
    "panic disorder",
    "social anxiety disorder",
    "specific phobia",
]

# ─── I/O paths ────────────────────────────────────────────────────────────────
try:                                    # running as a normal script
    here = pathlib.Path(__file__).resolve().parent
except NameError:                       # running in a notebook / REPL
    here = pathlib.Path(os.getcwd())    # fall back to the CWD

OUTPUT_FILE = here / "anxiety_papers.json"

# ─── PubMed helpers ───────────────────────────────────────────────────────────
def build_term(q, y_min, y_max, filt):
    return f"({q}) AND {filt} AND ({y_min}[PDAT] : {y_max}[PDAT])"

def esearch(term, retmax):
    p = dict(db="pubmed", term=term, retmode="json", retmax=retmax, sort="relevance")
    if API_KEY: p["api_key"] = API_KEY
    r = requests.get(ESEARCH, params=p, timeout=20)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]

def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def parse_abstract(article):
    """Return full abstract with section labels preserved"""
    parts = []
    for t in article.find_all("AbstractText"):
        label = t.get("Label") or t.get("NlmCategory")
        txt   = t.get_text(" ", strip=True)
        if label:
            parts.append(f"{label}: {txt}")
        else:
            parts.append(txt)
    return "\n".join(parts).strip() or "No abstract"

def efetch_batch(pmids):
    ids = ",".join(pmids)
    p = dict(db="pubmed", id=ids, retmode="xml")
    if API_KEY: p["api_key"] = API_KEY
    r = requests.get(EFETCH, params=p, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, PARSER)

    for art in soup.find_all("PubmedArticle"):
        pmid = art.PMID.text
        title = art.find("ArticleTitle")
        title = title.text.strip() if title else ""
        abstract = parse_abstract(art)
        journal = art.find("Title")
        journal = journal.text.strip() if journal else ""
        yr_tag  = art.find("PubDate") or art.find("DateCompleted")
        year    = yr_tag.Year.text if yr_tag and yr_tag.Year else ""
        authors=[]
        for au in art.find_all("Author"):
            fore = au.ForeName.text if au.ForeName else ""
            last = au.LastName.text if au.LastName else ""
            if fore or last:
                authors.append(f"{fore} {last}".strip())
        yield {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors
        }

# ─── scraping orchestrator ────────────────────────────────────────────────────
def scrape(max_sota, max_legacy, dry=0, verbose=True):
    today = datetime.date.today().year
    sota_filter   = "(systematic review[ptyp] OR randomized controlled trial[ptyp] OR guideline[ptyp] OR meta-analysis[ptyp])"
    legacy_filter = "review[ptyp]"

    wanted, seen = [], set()

    def collect(y_min, y_max, filt, quota):
        for q in QUERIES:
            term  = build_term(q, y_min, y_max, filt)
            pmids = esearch(term, quota)
            if verbose:
                print(f'Query "{q}" [{y_min}-{y_max}] → {len(pmids)} PMIDs')
            for group in chunk(pmids, BATCH):
                time.sleep(0.35)          # keep within NCBI rate limits
                for art in efetch_batch(group):
                    if art["pmid"] not in seen and art["abstract"]!="No abstract":
                        wanted.append(art)
                        seen.add(art["pmid"])
                    if dry and len(wanted) >= dry:
                        return

    collect(today-YEARS_SOTA+1, today, sota_filter,   max_sota)
    collect(today-YEARS_LEGACY,  today-YEARS_SOTA, legacy_filter, max_legacy)
    return wanted

# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Download recent SOTA + classic review anxiety papers and store JSON in ./dataset/.
            Examples:
              python anxiety_batch_scraper.py
              python anxiety_batch_scraper.py --dry 10
        """))
    ap.add_argument("--max-sota",   type=int, default=500)
    ap.add_argument("--max-legacy", type=int, default=350)
    ap.add_argument("--dry",        type=int, default=0, help="stop after N papers (debug)")
    args = ap.parse_args()

    papers = scrape(args.max_sota, args.max_legacy, args.dry)
    print(f"\nCollected {len(papers)} papers.")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved → {OUTPUT_FILE}")