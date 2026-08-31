"""Original web-scraping exercise."""

import requests
from bs4 import BeautifulSoup

DOC_URL = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"


def get_docs_data(doc_url: str) -> list[list[int | str]]:
    response = requests.get(doc_url, timeout=30)
    response.raise_for_status()
    content = BeautifulSoup(response.text, "html.parser").select_one("#contents")
    return [
        [int(cell[0].get_text(strip=True)), cell[1].get_text(strip=True), int(cell[2].get_text(strip=True))]
        for row in content.find("table").find_all("tr")[1:]
        if (cell := row.find_all("td"))
    ]


if __name__ == "__main__":
    print(get_docs_data(DOC_URL))

