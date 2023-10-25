from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


PRETRAINED_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
CONFIG = AutoConfig.from_pretrained(PRETRAINED_MODEL)
MODEL = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)


def preprocess_tweet(text:str) -> str:
    """
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    
    return " ".join(new_text)


def get_sentiment(text: str, use_tf: bool=False) -> str:
    """
    """
    text = preprocess_tweet(text)
    encoded_input = TOKENIZER(text, return_tensors='pt')
    output = MODEL(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    score_list = [score for score in scores]
    return np.array(score_list)
