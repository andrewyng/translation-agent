import json

import requests
from bs4 import BeautifulSoup


def extract_text(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract all text from the page
    text = soup.find_all("p")
    new_text = []
    for txt in text[2:]:
        try:
            filt = txt.get_text()
            print("------------")
            print(filt)
            filt = filt.split(")")[1:]
            filt = ")".join(filt)
            if filt != "":
                new_text.append({"text": filt})
        except:
            pass
    return new_text


if __name__ == "__main__":
    url = "https://www.deeplearning.ai/the-batch/data-points-issue-250/"
    text = extract_text(url)

    with open("data_points_samples.json", "w") as f:
        json.dump(text, f)

    print(text)
