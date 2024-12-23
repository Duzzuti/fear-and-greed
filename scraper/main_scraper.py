import sys
for path in sys.path:
    if path.endswith("\\fear-and-greed\\scraper"):
        sys.path.append(path.split("\\fear-and-greed\\scraper")[0] + "\\fear-and-greed")
        break
import utils

utils.fetch_all("data/", "2000-01-01")