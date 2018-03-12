# news-coverage
Uses visualisations and tf-idf to analyse news coverage.

Requires Python 3.  
Requires [this dataset](http://research.signalmedia.co/newsir16/index.html) to be downloaded and saved as sample-1M.jsonl in the directory.

#### Files

1. Run `get_data.py` to remove irrelevant articles from the dataset and convert it into a csv file so that it is more readable for Pandas.
2. Run `main.py` to create the following files:
    `VW_article_volD.png`  
        Histogram of articles per day.
    `VW_article_volH.png`  
        Histogram of articles per hour.
    `tfidf_words.csv`  
        Words on each day that have the highest tf-idf scores.
    `tfidf_scores.csv`  
        tf-idf scores of those words.
