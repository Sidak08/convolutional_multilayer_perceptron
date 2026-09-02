import requests
import pandas as pd
from io import StringIO

url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"

def groupCoord(table):
    rows = {}
    for x, char, y in table[
        ["x-coordinate", "Character", "y-coordinate"]
    ].itertuples(index=False, name=None):
        rows.setdefault(int(y), []).append((int(x), char))
    return {
        y: sorted(rows[y])
        for y in sorted(rows, reverse=True)
    }

def printCoord(rows):
    maxX, maxY, minY = max(x for points in rows.values() for x, _ in points), max(rows), min(rows)
    for y in range(maxY, minY - 1, -1):
        points = dict(rows.get(y, []))
        line = "".join(
            points.get(x, " ")
            for x in range(maxX + 1)
        )
        print(line)

def printDoc(link):
    response = requests.get(url, timeout=20).content.decode("utf-8")
    table = pd.read_html(StringIO(response), header=0)[0]
    printCoord(groupCoord(table))

printDoc(url)
