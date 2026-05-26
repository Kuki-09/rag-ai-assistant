import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def load_web(url: str):
    """Scrape a web page and return a LangChain Document."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL '{url}': {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return [
        Document(
            page_content=text,
            metadata={"source": url, "type": "web"},
        )
    ]