import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EvidenceBot | Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"

STOP = set("""
a an the is are was were be been do does did can could should would will
of in on for to with and or not what which who whom how why when where
its this that these those by from as at about into than then there their
they them have has had any some more most less least between
""".split())

HEADERS = {
    "User-Agent": "EvidenceBot/2.0 (biomedical literature research tool)"
}

# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #122c4b 0%, transparent 36%),
        radial-gradient(ellipse at 95% 15%, #103c3a 0%, transparent 30%),
        linear-gradient(145deg, #07111f 0%, #0b1728 55%, #091522 100%);
    color: #e8f0fa;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2c, #101e2d);
    border-right: 1px solid #24384b;
}

h1, h2, h3 {
    color: #f1f7ff !important;
}

p, label, li {
    color: #c3d2e3;
}

.hero {
    padding: 30px;
    border: 1px solid #294b61;
    border-radius: 22px;
    background: linear-gradient(
        125deg,
        rgba(20, 48, 75, 0.94),
        rgba(12, 49, 52, 0.83)
    );
    margin-bottom: 24px;
}

.eyebrow {
    color: #63e6d1;
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 800;
}

.hero-title {
    font-size: clamp(32px, 4vw, 50px);
    line-height: 1.15;
    font-weight: 850;
    color: #f3f8ff;
    margin: 12px 0;
}

.hero-subtitle {
    color: #c0d2e5;
    font-size: 16px;
    line-height: 1.7;
}

.metric {
    background: rgba(18, 36, 55, 0.92);
    border: 1px solid #294155;
    border-radius: 16px;
    padding: 18px;
    min-height: 112px;
}

.metric-label {
    font-size: 12px;
    color: #a7bdd1;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 30px;
    color: #68e4d0;
    font-weight: 800;
    margin-top: 8px;
}

.paper-card {
    background: linear-gradient(
        135deg,
        rgba(19, 38, 58, 0.97),
        rgba(12, 28, 44, 0.97)
    );
    border: 1px solid #294155;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0;
}

.paper-title {
    color: #f1f7ff;
    font-size: 19px;
    font-weight: 750;
    line-height: 1.5;
    margin-bottom: 10px;
}

.paper-meta {
    color: #70dfd1;
    font-size: 12px;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.paper-abstract {
    color: #c4d2e2;
    font-size: 14px;
    line-height: 1.8;
}

.source-chip {
    display: inline-block;
    background: #163a43;
    color: #8df1df;
    border: 1px solid #2c7777;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    margin: 3px 5px 3px 0;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #14b8a6, #0d9488);
    color: #061520;
    font-weight: 800;
    border: none;
    border-radius: 10px;
    min-height: 45px;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #5eead4, #2dd4bf);
    color: #061520;
    border: none;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #102236;
    border-color: #34516a;
}

hr {
    border-color: #24394c;
}

.small-note {
    font-size: 12px;
    color: #91a8be;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION AND MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def clean_query(question):
    words = re.findall(r"[A-Za-z0-9-]+", question.lower())
    useful = [w for w in words if w not in STOP and len(w) > 2]
    return " ".join(useful) or question.strip()


def safe_get(url, params=None, timeout=25):
    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def year_value(value):
    """Extract a publication year when one is available."""
    if value is None:
        return None

    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group()) if match else None


def make_paper(source, title, year=None, abstract="",
               url="", doi="", pmid="", paper_type="",
               open_access=False):
    title = str(title or "").strip()

    if not title:
        return None

    if isinstance(abstract, list):
        abstract = " ".join(str(x) for x in abstract)

    return {
        "source": source,
        "title": title,
        "year": year_value(year),
        "abstract": re.sub(r"\s+", " ", str(abstract or "")).strip(),
        "url": url or "",
        "doi": doi or "",
        "pmid": str(pmid or ""),
        "paper_type": paper_type or "",
        "open_access": bool(open_access),
        "score": None,
    }


def reconstruct_abstract(index):
    """Reconstruct an OpenAlex inverted-index abstract."""
    if not isinstance(index, dict) or not index:
        return ""

    positions = []
    for word, offsets in index.items():
        if isinstance(offsets, list):
            for offset in offsets:
                if isinstance(offset, int):
                    positions.append((offset, word))

    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


# =========================================================
# PUBMED
# =========================================================

def search_pubmed(question, limit):
    query = clean_query(question)

    response = safe_get(
        f"{PUBMED}/esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
            "sort": "relevance",
        },
    )

    ids = response.json()["esearchresult"]["idlist"]

    if not ids:
        return []

    response = safe_get(
        f"{PUBMED}/efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        },
    )

    root = ET.fromstring(response.content)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""

        title_element = article.find(".//ArticleTitle")
        title = (
            "".join(title_element.itertext())
            if title_element is not None else ""
        )

        abstract_parts = []
        for element in article.findall(".//Abstract/AbstractText"):
            text = "".join(element.itertext()).strip()
            if text:
                label = element.attrib.get("Label", "")
                if label:
                    text = f"{label}: {text}"
                abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        date = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate")
            or ""
        )

        doi = ""
        for identifier in article.findall(".//ArticleId"):
            if identifier.attrib.get("IdType") == "doi":
                doi = identifier.text or ""
                break

        paper = make_paper(
            source="PubMed",
            title=title,
            year=date,
            abstract=abstract,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            doi=doi,
            pmid=pmid,
        )

        if paper:
            papers.append(paper)

    return papers


# =========================================================
# EUROPE PMC
# =========================================================

def search_europe_pmc(question, limit):
    query = clean_query(question)

    response = safe_get(
        f"{EUROPE_PMC}/search",
        {
            "query": f'TITLE_ABS("{query}")',
            "format": "json",
            "pageSize": limit,
            "resultType": "core",
        },
    )

    papers = []

    for item in response.json().get("resultList", {}).get("result", []):
        pmid = item.get("pmid", "")
        doi = item.get("doi", "")
        source_id = item.get("id", "")

        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = (
                "https://europepmc.org/article/"
                f"{item.get('source', 'MED')}/{source_id}"
            )

        papers.append(make_paper(
            source="Europe PMC",
            title=item.get("title", ""),
            year=item.get("firstPublicationDate")
                 or item.get("pubYear"),
            abstract=item.get("abstractText", ""),
            url=url,
            doi=doi,
            pmid=pmid,
            paper_type=item.get("pubType", ""),
            open_access=item.get("isOpenAccess") == "Y",
        ))

    return [p for p in papers if p]


# =========================================================
# OPENALEX
# =========================================================

def search_openalex(question, limit):
    response = safe_get(
        OPENALEX,
        {
            "search": clean_query(question),
            "per-page": limit,
            "select": (
                "id,doi,title,publication_year,authorships,"
                "abstract_inverted_index,primary_location,"
                "open_access,type,cited_by_count"
            ),
        },
    )

    papers = []

    for item in response.json().get("results", []):
        doi = (item.get("doi") or "").replace("https://doi.org/", "")

        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        landing_page = location.get("landing_page_url") or ""

        if doi:
            url = f"https://doi.org/{doi}"
        else:
            url = landing_page or item.get("id", "")

        authors = item.get("authorships") or []
        author_names = [
            a.get("author", {}).get("display_name", "")
            for a in authors[:5]
        ]
        author_text = ", ".join(x for x in author_names if x)

        paper = make_paper(
            source="OpenAlex",
            title=item.get("title", ""),
            year=item.get("publication_year"),
            abstract=reconstruct_abstract(
                item.get("abstract_inverted_index")
            ),
            url=url,
            doi=doi,
            paper_type=item.get("type", ""),
            open_access=(
                (item.get("open_access") or {}).get("is_oa") is True
            ),
        )

        if paper:
            paper["authors"] = author_text
            paper["citations"] = item.get("cited_by_count", 0)
            paper["journal"] = source.get("display_name", "")
            papers.append(paper)

    return papers


# =========================================================
# CROSSREF
# =========================================================

def search_crossref(question, limit):
    response = safe_get(
        CROSSREF,
        {
            "query.bibliographic": clean_query(question),
            "rows": limit,
            "select": (
                "DOI,title,author,published,published-print,"
                "published-online,container-title,type,URL"
            ),
        },
    )

    papers = []

    for item in response.json().get("message", {}).get("items", []):
        titles = item.get("title") or []
        title = titles[0] if titles else ""

        date_parts = (
            item.get("published")
            or item.get("published-print")
            or item.get("published-online")
            or {}
        ).get("date-parts", [[]])

        year = date_parts[0][0] if date_parts and date_parts[0] else None
        doi = item.get("DOI", "")

        paper = make_paper(
            source="Crossref",
            title=title,
            year=year,
            abstract="",
            url=f"https://doi.org/{doi}" if doi else item.get("URL", ""),
            doi=doi,
            paper_type=item.get("type", ""),
        )

        if paper:
            papers.append(paper)

    return papers


# =========================================================
# COMBINE AND DEDUPLICATE RECORDS
# =========================================================

def merge_papers(groups):
    merged = {}
    source_errors = []

    for source, papers, error in groups:
        if error:
            source_errors.append(f"{source}: {error}")

        for paper in papers:
            if not paper:
                continue

            doi = paper.get("doi", "").lower().strip()
            pmid = paper.get("pmid", "").strip()
            title_key = re.sub(
                r"[^a-z0-9]+", "",
                paper["title"].lower()
            )

            key = (
                f"doi:{doi}" if doi
                else f"pmid:{pmid}" if pmid
                else f"title:{title_key}"
            )

            if key not in merged:
                merged[key] = paper
                continue

            existing = merged[key]

            # Preserve all source labels for duplicate records.
            labels = existing.get("sources", [existing["source"]])
            if paper["source"] not in labels:
                labels.append(paper["source"])
            existing["sources"] = labels

            # Prefer a record that contains an abstract.
            if not existing.get("abstract") and paper.get("abstract"):
                existing["abstract"] = paper["abstract"]

            # Preserve a usable URL.
            if not existing.get("url") and paper.get("url"):
                existing["url"] = paper["url"]

            # Preserve missing metadata when another source supplies it.
            for field in ("year", "doi", "pmid", "paper_type"):
                if not existing.get(field) and paper.get(field):
                    existing[field] = paper[field]

    return list(merged.values()), source_errors


# =========================================================
# SEMANTIC EVIDENCE RANKING
# =========================================================

def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 30
    ]


def rank_papers(question, papers, model, threshold):
    evidence_rows = []

    for paper_index, paper in enumerate(papers):
        for sentence in split_sentences(paper.get("abstract", "")):
            evidence_rows.append({
                "paper_index": paper_index,
                "sentence": sentence,
            })

    if not evidence_rows:
        return papers, []

    sentences = [row["sentence"] for row in evidence_rows]

    sentence_vectors = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_vector = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = sentence_vectors @ question_vector

    best_by_paper = {}

    for row, score in zip(evidence_rows, scores):
        score = float(score)
        index = row["paper_index"]

        if (
            index not in best_by_paper
            or score > best_by_paper[index]["score"]
        ):
            best_by_paper[index] = {
                "sentence": row["sentence"],
                "score": score,
            }

    for index, result in best_by_paper.items():
        papers[index]["score"] = result["score"]

    ranked = sorted(
        best_by_paper.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    evidence = []

    for index, result in ranked:
        if result["score"] >= threshold:
            evidence.append({
                "paper": papers[index],
                "sentence": result["sentence"],
                "score": result["score"],
            })

    return papers, evidence


# =========================================================
# USER INTERFACE
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Biomedical Research Intelligence</div>
    <div class="hero-title">Explore the evidence.</div>
    <div class="hero-subtitle">
        Discover biomedical literature across multiple research databases.
        Retrieve abstracts, compare sources, and explore original publications
        in one workspace.
    </div>
    <br>
    <span class="source-chip">PUBMED</span>
    <span class="source-chip">EUROPE PMC</span>
    <span class="source-chip">OPENALEX</span>
    <span class="source-chip">CROSSREF</span>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔎 Ask a biomedical research question")

with st.form("research_form"):
    question = st.text_input(
        "Research question",
        placeholder=(
            "e.g. Does vitamin D deficiency associate "
            "with type 2 diabetes?"
        ),
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_sources = st.multiselect(
            "Search databases",
            ["PubMed", "Europe PMC", "OpenAlex", "Crossref"],
            default=["PubMed", "Europe PMC", "OpenAlex", "Crossref"],
        )

    with col2:
        limit = st.selectbox(
            "Records per source",
            [5, 10, 15, 20],
            index=1,
        )

    submitted = st.form_submit_button(
        "🔬 Search Evidence",
        type="primary",
        use_container_width=True,
    )

with st.sidebar:
    st.markdown("## ⚙️ Research settings")

    min_year = st.number_input(
        "Earliest publication year",
        min_value=1900,
        max_value=2100,
        value=1900,
        step=1,
    )

    max_year = st.number_input(
        "Latest publication year",
        min_value=1900,
        max_value=2100,
        value=2100,
        step=1,
    )

    threshold = st.slider(
        "Sentence relevance threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help=(
            "This is a semantic similarity threshold, "
            "not a measure of scientific certainty."
        ),
    )

    open_access_only = st.checkbox("Open-access records only")

    st.divider()

    st.markdown("### About EvidenceBot")
    st.caption(
        "A multi-source biomedical literature discovery prototype. "
        "Search results depend on each database's coverage and API."
    )

    st.caption(
        "Sentence similarity identifies relevant text; it does not "
        "verify causality, study quality, or whether a claim is true."
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a biomedical research question.")
        st.stop()

    if not selected_sources:
        st.warning("Select at least one database.")
        st.stop()

    if min_year > max_year:
        st.error("The earliest year must not exceed the latest year.")
        st.stop()

    search_functions = {
        "PubMed": search_pubmed,
        "Europe PMC": search_europe_pmc,
        "OpenAlex": search_openalex,
        "Crossref": search_crossref,
    }

    groups = []

    progress = st.progress(0)
    status = st.empty()

    for number, source in enumerate(selected_sources, start=1):
        status.info(f"Searching {source}...")

        try:
            results = search_functions[source](question, limit)
            groups.append((source, results, None))
        except Exception as error:
            groups.append((source, [], str(error)))

        progress.progress(number / len(selected_sources))
        time.sleep(0.1)

    papers, source_errors = merge_papers(groups)

    # Apply publication-year and access filters.
    filtered = []

    for paper in papers:
        year = paper.get("year")

        if year is not None and not (min_year <= year <= max_year):
            continue

        if open_access_only and not paper.get("open_access", False):
            continue

        filtered.append(paper)

    status.empty()
    progress.empty()

    if source_errors:
        with st.expander("⚠️ Some sources could not be searched"):
            for error in source_errors:
                st.warning(error)

    if not filtered:
        st.warning(
            "No records matched the selected filters. "
            "Try a broader question, more sources, or a wider year range."
        )
        st.stop()

    abstract_count = sum(
        bool(paper.get("abstract")) for paper in filtered
    )

    # Load the model only when abstracts need to be ranked.
    evidence = []

    if abstract_count:
        with st.spinner(
            "Loading the semantic model and ranking available abstracts..."
        ):
            model = load_model()
            filtered, evidence = rank_papers(
                question, filtered, model, threshold
            )

    filtered.sort(
        key=lambda paper: (
            paper.get("score") is not None,
            paper.get("score") or 0,
        ),
        reverse=True,
    )

    st.session_state["papers"] = filtered
    st.session_state["evidence"] = evidence
    st.session_state["search_question"] = question
    st.session_state["source_errors"] = source_errors


# =========================================================
# DISPLAY SEARCH RESULTS
# =========================================================

if "papers" in st.session_state:
    papers = st.session_state["papers"]
    evidence = st.session_state.get("evidence", [])
    question = st.session_state.get("search_question", "")

    st.divider()
    st.markdown("## 📊 Research overview")

    source_count = len({
        source
        for paper in papers
        for source in paper.get("sources", [paper["source"]])
    })

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "Records found", len(papers)),
        (c2, "With abstracts", sum(bool(p["abstract"]) for p in papers)),
        (c3, "Sources represented", source_count),
        (c4, "Relevant sentences", len(evidence)),
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(
                f'<div class="metric">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("## 🧠 Relevant abstract sentences")

    if evidence:
        st.caption(
            "Ranked by semantic similarity to your question. "
            "These are candidate passages, not verified conclusions."
        )

        for item in evidence[:10]:
            paper = item["paper"]

            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-meta">'
                f'{paper["source"].upper()}'
                f' &nbsp;•&nbsp; '
                f'{paper.get("year") or "Year unavailable"}'
                f' &nbsp;•&nbsp; '
                f'Similarity: {item["score"]:.2f}'
                f'</div>'
                f'<div class="paper-abstract">'
                f'{item["sentence"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if paper.get("url"):
                st.markdown(f"[View original publication ↗]({paper['url']})")

    else:
        st.info(
            "No abstract sentences met the selected similarity threshold. "
            "Some records may have metadata but no accessible abstract. "
            "This does not mean no scientific evidence exists."
        )

    st.markdown("## 📚 Discovered publications")

    search_in_titles = st.text_input(
        "Filter results by title",
        placeholder="Search within retrieved titles...",
        key="title_filter",
    )

    display_papers = [
        paper for paper in papers
        if search_in_titles.lower() in paper["title"].lower()
    ]

    for paper in display_papers:
        sources = paper.get("sources", [paper["source"]])
        source_labels = " · ".join(sources)

        st.markdown(
            f'<div class="paper-card">'
            f'<div class="paper-meta">'
            f'{source_labels.upper()}'
            f' &nbsp;•&nbsp; '
            f'{paper.get("year") or "Year unavailable"}'
            f'</div>'
            f'<div class="paper-title">{paper["title"]}</div>'
            f'<div class="paper-abstract">'
            f'{paper.get("abstract") or "No abstract available in this record."}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        meta = []

        if paper.get("doi"):
            meta.append(f'DOI: {paper["doi"]}')

        if paper.get("pmid"):
            meta.append(f'PMID: {paper["pmid"]}')

        if paper.get("paper_type"):
            meta.append(f'Type: {paper["paper_type"]}')

        if meta:
            st.caption(" · ".join(meta))

        if paper.get("url"):
            st.markdown(f"[Open publication or source record ↗]({paper['url']})")

        st.divider()

    # Export retrieved metadata.
    if papers:
        import pandas as pd

        export_rows = []

        for paper in papers:
            export_rows.append({
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "source": ", ".join(
                    paper.get("sources", [paper["source"]])
                ),
                "doi": paper.get("doi", ""),
                "pmid": paper.get("pmid", ""),
                "url": paper.get("url", ""),
                "abstract": paper.get("abstract", ""),
                "similarity_score": paper.get("score"),
            })

        csv_data = pd.DataFrame(export_rows).to_csv(index=False)

        st.download_button(
            "⬇️ Export results as CSV",
            data=csv_data,
            file_name="evidencebot_results.csv",
            mime="text/csv",
        )

st.divider()

st.markdown(
    '<div class="small-note">'
    'EvidenceBot is a literature discovery and research assistance tool. '
    'Database coverage, abstracts, indexing, and API availability vary. '
    'Similarity scores are not probabilities, evidence grades, '
    'or measures of study quality. Always verify claims in the original papers.'
    '</div>',
    unsafe_allow_html=True,
)