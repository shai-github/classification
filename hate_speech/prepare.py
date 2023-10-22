import pandas as pd

from utils.clean import clean_text
from utils.embed import embed_text
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


SCALER = StandardScaler()
ROS = RandomOverSampler()


def prepare_dataframe() -> pd.DataFrame:
    """
    Read data from csv file and prepare a dataframe with
        Before running, make sure that data is correctly placed
        in the data folder as `hate_speech/data/hsol.csv`
    :return: dataframe with adjusted columns and labels
    """
    df = pd.read_csv('hate_speech/data/hsol.csv')

    # only keep tweet text and label class
    df = df[['tweet', 'class']]

    # change labels for different logic
    # 0 for neither hate speech nor offensive language
    # 1 for offensive language
    # 2 for hate speech
    df['class'] = df['class'].apply(lambda x: 0 if x == 2 else 2 if x == 0 else 1)

    return df


def weight_data():
    """
    
    """
    pass


def clean_and_embed(df: pd.DataFrame):
    """
    """
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df['embeddings'] = df['clean_tweet'].apply(embed_text)