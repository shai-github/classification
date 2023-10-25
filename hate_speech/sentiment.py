import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


PRETRAINED_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
CONFIG = AutoConfig.from_pretrained(PRETRAINED_MODEL)
MODEL = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)


def preprocess_tweet(text:str) -> str:
    """
    Preprocesses the text by removing mentions and urls
    :param text: tweet text
    :return: preprocessed tweet text
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    
    return " ".join(new_text)


def get_sentiment(text: str) -> list:
    """
    Retrieves the sentiment score for a given tweet text
    :param text: tweet text
    :return: list of sentiment scores
    """
    text = preprocess_tweet(text)
    encoded_input = TOKENIZER(text, return_tensors='pt')
    output = MODEL(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    score_list = [score for score in scores]
    return np.array(score_list)
