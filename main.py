import pprint, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# initiate plotting settings
attributes = {
    'axes.facecolor' : '#f0e6f2'
}
sns.set(context='paper',style='darkgrid',rc=attributes)
sns.set_palette(sns.cubehelix_palette(5,reverse=True))#sns.diverging_palette(270,320,center='dark',n=7))

def import_data():
    """Retrieve VW articles from csv file."""
    df = pd.read_csv('vw_articles.csv')
    # print(df.shape)
    df.published = pd.to_datetime(df.published, format='%Y-%m-%dT%H:%M:%SZ')
    # print(df[df.published.isnull()].shape[0],'have missing dates')
    df = df[df.published.notnull()]
    # print(df.head())
    return df

def analyse_words():
    """Find most relevant words for each day."""
    df = import_data()
    df.time_period = df.published.dt.to_period('D')
    # get all content from one day into the appropriate entry in df_days
    time_periods = df.time_period.unique()
    time_periods = np.sort(time_periods)
    time_periods = time_periods[10:]
    df_days = pd.Series([0]*len(time_periods),index=time_periods)
    for day in time_periods:
        df_tmp = df[df.time_period == day]
        day_content = ' '.join(df_tmp.content.values)
        # if day == time_periods[3]:
            # df_tmp = df_tmp[['title','published','source','media-type']]
            # df_tmp = df_tmp.sort_values('published')
            # print(df_tmp)
        df_days[day] = day_content

    # fill in dates that don't have any articles with a 0     
    empty_date = df_days.index[0]
    while empty_date <= df_days.index[-1]:
        if empty_date not in df_days.index:
            df_days.loc[empty_date] = ''
            df_days = df_days.sort_index()
            time_periods = np.append(time_periods,empty_date)
        empty_date += 1
    time_periods = np.sort(time_periods)

    # prepare df_days for tf-idf
    initialisation = ' '.join(df_days.values)

    documents = iter(df_days.values)
    # do tf-idf
    stop_words = stopwords.words('english')
    # Uncomment to add words to the stopwords list
    # stop_words.extend(['volkswagen','car','cars','emissions','scandal'])
    tfidf = TfidfVectorizer(stop_words=stop_words)#, sublinear_tf=True)
    sparse = tfidf.fit_transform(documents)
    features = tfidf.get_feature_names()
    # set up for plot of tfidf over time
    chosen_wrds = ['EPA','Winterkorn','Mueller','Volkswagen','Nitrogen']
    tfidf_plot = pd.DataFrame(columns=chosen_wrds,index=time_periods)
    chosen_ind = [features.index(wrd.lower()) for wrd in chosen_wrds]
    n_best = 20
    words_df = pd.DataFrame(columns=time_periods,index=np.arange(n_best))
    scores_df = pd.DataFrame(columns=time_periods,index=np.arange(n_best))
    for rowid in range(sparse.shape[0]):
        row = np.squeeze(sparse[rowid].toarray())
        # Retrieve top 20 words and their tfidf scores
        # best_ids = np.argsort(row)[::-1][:n_best]
        # best_features,tfidf_scores = zip(*[(features[i],row[i]) for i in best_ids])
        # scores_df[time_periods[rowid]] = tfidf_scores
        # if tfidf_scores[0] != 0:
        #     words_df[time_periods[rowid]] = best_features

        # add to data for tfidf plot
        tfidf_plot.loc[time_periods[rowid]] = row[chosen_ind]
    print('\n\nTfidf values for plot:')
    print(tfidf_plot)
    fig,ax = plt.subplots()
    for column in tfidf_plot:
        tfidf_plot[column].plot(kind='line',linewidth=2,ax=ax)
    ax.legend()
    ax.set_xticklabels(time_periods)
    ax.xaxis.set_minor_locator(plticker.NullLocator())
    ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=83,ha='center')
    plt.title('tf-idf Scores')
    plt.savefig('tfidf_plot.png')
    words_df.to_csv('tfidf_words.csv')
    scores_df.to_csv('tfidf_scores.csv')

def plot_vol(time_period):
    """Plot article volume"""
    if time_period == 'H':
        params = {'start' : 0, 'step' : 24, 'ylim' : 80, 'bottom' : 0.3, 'period_name': 'Hour', 'figsize':[16,4]}
    if time_period == 'D':
        params = {'start' : 2, 'step' : 3, 'ylim' : 700, 'bottom' : 0.2, 'period_name': 'Day', 'figsize':[8,6.5]}
        # print(plt.rcParams['font.size'])
        # plt.rcParams['font.size'] = 15
        # print(plt.rcParams['font.size'])

    df = import_data()
    # allocate articles to time periods for plotting
    df = df.published.groupby(df.published.dt.to_period(time_period)).count()
    # fill in dates that don't have any articles with a 0
    empty_date = df.index[0]
    while empty_date <= df.index[-1]:
        if empty_date not in df.index:
            df.loc[empty_date] = 0
            df = df.sort_index()
        empty_date += 1
    if time_period == 'H':
        df = df[df.index >= datetime.datetime(2015,9,18)]

    # plot
    fig,ax = plt.subplots(figsize = params['figsize'])
    df.plot(kind='bar',ax=ax, color='purple')
    plt.title("Number of Volkswagen 'emissions' or 'scandal' Articles per "+params['period_name'],fontsize=12)
    plt.xlabel("")
    fig.subplots_adjust(bottom=params['bottom'])
    indices = list(range(params['start'],df.shape[0],params['step']))
    plt.xticks(indices,df.index[indices],rotation=83,ha='center')
    plt.ylim(0,params['ylim'])
    plt.savefig('VW_article_vol'+time_period+'.png')

def plot_hours():
    time_period='H'
    plot_vol(time_period=time_period)

def plot_days():    
    time_period='D'
    plot_vol(time_period=time_period)

if __name__ == '__main__':
    # plot_days()
    # plot_hours()
    analyse_words()
