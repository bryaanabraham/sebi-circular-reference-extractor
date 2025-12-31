import os
import re
import time
import hashlib
import requests
import pdfplumber
from collections import defaultdict
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.sebi.gov.in"
AJAX_URL = BASE + "/sebiweb/ajax/home/getnewslistinfo.jsp"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": BASE + "/sebiweb/home/HomeAction.do"
}

PDF_DIR = "pdfs"
TXT_DIR = "txts"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

def safe_filename(title, max_len = 150):
    '''
    Function to prevent any errors in file name while saving pdf/txt files
    '''
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()

    if len(title) > max_len:
        suffix = hashlib.md5(title.encode()).hexdigest()[:8]
        title = f"{title[:max_len]}_{suffix}"

    return title


def batch_iterable(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def get_pdf_metadata(session, url, fallback_id):
    '''
    Function to get the pdf download link.
    1. If an iframe is detected in the page i.e. there is a pdf in the page the we extract the link and save if to a dictionary
    2. If the circular is directly in a text format, the text is saved locally to a txt file and the file name is saved to the dictionary
    Return a single entry for the dictionary for each link
    '''

    try:
        res = session.get(url, timeout=30)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.select_one(
            "section.department-slider.news_main.news-detail-slider h1"
        )
        title = title_tag.get_text(strip=True) if title_tag else f"sebi_{fallback_id}"
        title = safe_filename(title)

        iframe = soup.find("iframe", src=True)
        if iframe:
            src = iframe["src"]
            parsed = urlparse(urljoin(url, src))
            pdf_url = parse_qs(parsed.query).get("file", [None])[0]

            if pdf_url:
                if not pdf_url.startswith("http"):
                    pdf_url = urljoin(BASE + "/", pdf_url)

                return {
                    "type": "pdf",
                    "title": title,
                    "source": pdf_url
                }

        soup = BeautifulSoup(res.text, "lxml")
        content = soup.find("div", class_="table-scrollable")

        if not content:
            return None

        text = re.sub(r'\n\s*\n+', '\n\n',
                      content.get_text(separator="\n")).strip()

        path = os.path.join(TXT_DIR, f"{title}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        return {
            "type": "text",
            "title": title,
            "source": path
        }

    except Exception as e:
        print("Metadata error:", url, e)
        return None


def collect_circular_links(session, pages=111):
    '''
    To scrape the pdf from the SEB website the data from pages 1 to 100 needs to be accessed.
    This data is scraped using this function which finds the html code for the page with individual circulars on them
    Returns the complete list of links for all pages with circulars
    '''
    links = []

    for page in tqdm(range(pages), desc="Fetching pages"):
        payload = {
            "nextValue": "1",
            "next": "n",
            "doDirect": str(page),
            "sid": "1",
            "ssid": "7",
            "smid": "0",
            "search": ""
        }

        r = session.post(AJAX_URL, data=payload, headers=HEADERS)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                href.startswith("https")
                and "index.html" not in href
                and "legal.html" not in href
            ):
                links.append(href)

        time.sleep(0.4)

    return list(set(links))


def download_pdf(url, filename):
    # Fucntion to download a single pdf from the link
    try:
        path = os.path.join(PDF_DIR, filename)
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    except Exception as e:
        print("Download failed:", filename, e)


def download_all_pdfs(pdf_items):
    # function to download pdfs in batches
    tasks = []
    seen = defaultdict(int)

    for item in pdf_items:
        if item["type"] != "pdf":
            continue

        base = safe_filename(item["title"])
        seen[base] += 1

        name = f"{base}_{seen[base]}.pdf" if seen[base] > 1 else f"{base}.pdf"
        tasks.append((item["source"], name))

    for batch in batch_iterable(tasks, 10):
        with ThreadPoolExecutor(max_workers=5) as exe:
            futures = [exe.submit(download_pdf, u, n) for u, n in batch]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
        time.sleep(1)


def pdf_to_text(pdf_file):
    # Fucntion to extract text from pdf and store it in a txt
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n\n"

        out = os.path.join(
            TXT_DIR,
            os.path.basename(pdf_file).replace(".pdf", ".txt")
        )

        with open(out, "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e:
        print("PDF parse error:", pdf_file, e)


if __name__ == "__main__":
    session = requests.Session()
    session.headers.update(HEADERS)

    session.get(
        BASE + "/sebiweb/home/HomeAction.do",
        params={"doListing": "yes", "sid": "1", "ssid": "7", "smid": "0"}
    )

    # Collect html links to all the pages containing the SEB circulars
    links = collect_circular_links(session)
    print("Found", len(links), "circular pages")

    # Extract the links for individual pdf/txt files from the html links
    items = []
    for idx, link in enumerate(tqdm(links, desc="Extracting metadata")):
        meta = get_pdf_metadata(session, link, idx)
        if meta:
            items.append(meta)

    # fall back in case of any failure to prevent repitition of heavy tasks
    with open("all_pdfs.txt", "w") as f:
        f.write(str(items))

    download_all_pdfs(items)

    # Extract text from all the pdfs and save it to a txt
    for file in tqdm(os.listdir(PDF_DIR), desc="Extracting text"):
        pdf_to_text(os.path.join(PDF_DIR, file))