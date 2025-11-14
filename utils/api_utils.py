import requests

def fetch_semantic_scholar(query, limit=3):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query_with_years = f"{query} year:2024-2025"

    params = {
        "query": query_with_years,
        "limit": limit,
        "fields": "title,abstract,year,url"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])
    else:
        print("Error fetching papers:", response.status_code, response.text)
        return []
