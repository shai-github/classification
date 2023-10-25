import pandas as pd
import numpy as np

from utils.clean import clean
from utils.embed import embed
from hate_speech.sentiment import get_sentiment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def prepare_dataframe(make_binary: bool = False) -> pd.DataFrame:
    """
    Read data from csv file and prepare a dataframe with
        Before running, make sure that data is correctly placed
        in the data folder as `hate_speech/data/hsol.csv`
    :param make_binary: whether to make the labels binary
        This means hate speech and offensive language are
        combined into one class for binary classification
    :return: dataframe with adjusted columns and labels
    """
    df = pd.read_csv('hate_speech/data/hsol.csv')

    # only keep tweet text and label class
    df = df[['tweet', 'class']]

    # change labels for different logic
    # 0 for neither hate speech nor offensive language
    # 1 for offensive language
    # 2 for hate speech
    if make_binary:
        df['class'] = df['class'].apply(lambda x: 0 if x == 2 else 1)
    else:
        df['class'] = df['class'].apply(lambda x: 0 if x == 2 else 2 if x == 0 else 1)

    return df


def weight_data(df: pd.DataFrame) -> dict:
    """
    Returns weights for each class in the dataframe based on the
        number of samples in each class - this is done to account
        for the imbalance in the dataset
    :param df: dataframe with labels
    :return: dictionary with weights for each class
    """
    count_list = [count for count in np.bincount(df['class'])]
    total = sum(count_list)

    return {
        i: (1 / count_type) * (total) / len(count_list) 
        for i, count_type in enumerate(count_list)
    }


def balance_by_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the dataframe by sampling the same number of samples
        from each class
    :param df: input dataframe
    :return: balanced dataframe
    """
    df_list = []
    min_count = min(np.bincount(df['class']))

    for i in range(3):
        df_list.append(
            df[df['class'] == i].sample(n=min_count, replace=False)
        )

    return pd.concat(df_list)


def merge_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the sentiment scores with the dataframe
    :param df: dataframe with embeddings
    :return: dataframe with sentiment scores and embeddings
    """    
    scale_df = pd.DataFrame()
    scale_df['embeddings'] = df['sentiment'].apply(lambda x: x.tolist()) + df['embeddings'].apply(lambda x: x.tolist())
    scale_df['class'] = df['class']

    return scale_df


def split_data(df: pd.DataFrame, use_sentiment: bool = False) -> dict:
    """
    Splits the dataframe into train and test sets
    :param df: dataframe with tweets and labels
    :return: dictionary with train and test sets
    """
    if use_sentiment:
        scale_df = merge_sentiment(df)
    else:
        scale_df = df.copy(deep=True)

    X_train, X_test, y_train, y_test = train_test_split(
        scale_df['embeddings'].to_list(), 
        scale_df['class'].to_list(), 
        test_size=0.2, 
        random_state=42
    )
    
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def clean_and_embed(df: pd.DataFrame, sample: int = 0, include_sentiment: bool = False) -> pd.DataFrame:
    """
    Cleans text and embeds it using the pretrained model
    :param df: dataframe with tweets
    :param sample: number of samples to use
    :param convert_to_np: whether to convert embeddings to numpy arrays
    :return: dataframe with cleaned tweets and embeddings
    """
    # create a deep copy of the dataframe
    if sample > 0:
        embed_df = df.sample(n=sample, random_state=42).copy(deep=True) 
    else:
        embed_df = df.copy(deep=True)

    # clean text and embed it
    embed_df['clean_tweet'] = embed_df['tweet'].apply(clean)
    embed_df['embeddings'] = embed_df['clean_tweet'].apply(embed)

    if include_sentiment:
        embed_df['sentiment'] = embed_df['tweet'].apply(get_sentiment)

    return embed_df
