# pubmed_scraper.py
import requests
from bs4 import BeautifulSoup
import json
import time
import random
import sys

# Check for required dependencies
try:
    import lxml
    PARSER = 'xml'
except ImportError:
    print("Warning: lxml not found. Using html.parser instead (less reliable for PubMed XML).")
    print("Consider installing lxml with: pip install lxml")
    PARSER = 'html.parser'

def search_pubmed(query, max_results=100):
    """Search PubMed for articles related to the query and return PMIDs."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        data = response.json()
        return data["esearchresult"]["idlist"]
    except requests.exceptions.RequestException as e:
        print(f"Error searching PubMed: {e}")
        return []
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing PubMed search results: {e}")
        return []

def get_abstract(pmid):
    """Fetch the abstract for a given PMID."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, PARSER)

        # Extract article details
        title_elem = soup.find('ArticleTitle')
        title = title_elem.text if title_elem else "No title available"

        abstract_elem = soup.find('AbstractText')
        if not abstract_elem:
            # Try alternative structure
            abstract_elems = soup.find_all('AbstractText')
            if abstract_elems:
                abstract = " ".join([elem.text for elem in abstract_elems])
            else:
                abstract = "No abstract available"
        else:
            abstract = abstract_elem.text

        journal_elem = soup.find('Title')
        journal = journal_elem.text if journal_elem else "No journal available"

        year_elem = soup.find('PubDate')
        year = "No year available"
        if year_elem:
            year_text = year_elem.find('Year')
            if year_text:
                year = year_text.text

        authors = []
        author_list = soup.find('AuthorList')
        if author_list:
            for author in author_list.find_all('Author'):
                last_name = author.find('LastName')
                last = last_name.text if last_name else ""

                fore_name = author.find('ForeName')
                fore = fore_name.text if fore_name else ""

                if last or fore:
                    authors.append(f"{fore} {last}".strip())

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return None
    except Exception as e:
        print(f"Error processing PMID {pmid}: {e}")
        return None

def main():
    # Search terms related to genetic mutations and cellular function
    queries = [
        "genetic mutation cellular function",
        "gene mutations AND cell signaling",
        "DNA mutation AND protein function",
        "mutation impact AND cellular pathways",
        "genetic variants AND cell metabolism"
    ]

    all_articles = []
    total_attempted = 0
    total_retrieved = 0

    for query in queries:
        print(f"Searching for: {query}")
        pmids = search_pubmed(query, max_results=25)  # 25 results per query = 125 total
        print(f"Found {len(pmids)} results for query: {query}")

        for pmid in pmids:
            total_attempted += 1
            print(f"Retrieving PMID: {pmid} ({total_retrieved}/{total_attempted})")

            article = get_abstract(pmid)
            if article and article["abstract"] != "No abstract available":
                all_articles.append(article)
                total_retrieved += 1
                print(f"Retrieved article: {article['title'][:50]}...")
            else:
                print(f"Skipped article (no abstract)")

            # Be nice to the NCBI servers
            time.sleep(random.uniform(1, 3))

    # Remove duplicates by PMID
    unique_articles = {article["pmid"]: article for article in all_articles}
    final_articles = list(unique_articles.values())

    # Save as JSON (structured format)
    with open("biomedical_abstracts.json", "w", encoding="utf-8") as f:
        json.dump(final_articles, f, indent=2)

    # Also save as text (for simple processing)
    with open("biomedical_abstracts.txt", "w", encoding="utf-8") as f:
        for article in final_articles:
            f.write(f"PMID: {article['pmid']}\n")
            f.write(f"TITLE: {article['title']}\n")
            f.write(f"AUTHORS: {', '.join(article['authors'])}\n")
            f.write(f"JOURNAL: {article['journal']} ({article['year']})\n")
            f.write(f"ABSTRACT:\n{article['abstract']}\n\n")
            f.write("-" * 80 + "\n\n")

    print(f"Saved {len(final_articles)} articles to files")
    print(f"JSON file: biomedical_abstracts.json")
    print(f"Text file: biomedical_abstracts.txt")

if __name__ == "__main__":
    main()