from newsapi import NewsApiClient
import pandas as pd
newsapi = NewsApiClient(api_key='a96eabfb357e4ea8b342c583297df626')
topics = ['economy', 'finance', 'stocks', 'investment']
def get_headlines_to_csv(topics):
    all_headlines = []
    for topic in topics:
        results = newsapi.get_everything(
            q =topic,
            language='en',
            domains='wsj.com,bloomberg.com',
            sort_by='relevancy'
        )
        for article in results['articles']:
            all_headlines.append(article['title'])

    return all_headlines
headlines = get_headlines_to_csv(topics)
headlines_df = pd.DataFrame(headlines, columns=['headlines'])
headlines_df.to_csv('apiheadlines.csv', index=False)