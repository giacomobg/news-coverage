import jsonlines, pprint, datetime
import pandas as pd

def articles2csv():
    """Open jsonlines file and turn it into an iterator."""
    # article keys: content, id, media-type, published, source, title
    columns = ['content','id','media-type','published','source','title']
    df = pd.DataFrame(columns=columns)
    with jsonlines.open('sample-1M.jsonl') as reader:
        for counter,artc in enumerate(reader):
            if 'Volkswagen' in artc['content']:
                if ('scandal' in artc['content']) or ('emissions' in artc['content']):
                    s = pd.Series([artc['content'],artc['id'],artc['media-type'],artc['published'],artc['source'],artc['title']],index=columns)
                    df = df.append(s,ignore_index=True)
    df.to_csv('vw_articles.csv',index=False)

if __name__ == '__main__':
    articles2csv()