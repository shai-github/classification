import pandas as pd
import numpy as np

from utils.clean import clean
from utils.embed import embed
from sklearn.model_selection import train_test_split


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
    n_count, o_count, h_count = np.bincount(df['class'])
    total = n_count + o_count + h_count

    return {
        i: (1 / count_type) * (total) / 3.0 
        for i, count_type in enumerate([n_count, o_count, h_count])
    }


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Splits the dataframe into train and test sets
    :param df: dataframe with tweets and labels
    :param test_size: size of the test set
    :return: dictionary with train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['embeddings'].to_list(), 
        df['class'].to_list(), 
        test_size=test_size, 
        random_state=42
    )
    
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def clean_and_embed(df: pd.DataFrame, sample: int = 0, use_sbert: bool = False, convert_to_np: bool = False) -> pd.DataFrame:
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
    embed_df['embeddings'] = embed_df['clean_tweet'].apply(embed, use_sbert=use_sbert)

    # if using a method that requires numpy arrays
    # convert the tensor embeddings to numpy arrays
    if convert_to_np:
        embed_df['embeddings'] = embed_df['embeddings'].apply(lambda x: x.detach().numpy())

    return embed_df
