import re
import html
import string
import unicodedata
import contractions

from typing import List, Union, Iterable
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer(r'\w+')
LEMMATIZER = WordNetLemmatizer()

HTML = re.compile(r"<[^>]*>")
HASHTAG = re.compile(r"#(\w+)")
EMOJIS = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF" 
    u"\U0001F680-\U0001F6FF" 
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", 
    flags=re.UNICODE
)


class NoText(Exception):
    """Error raised if text is empty or null"""


def _good_token(token: str) -> bool:
    """
    Returns False if a token is invalid, returns True otherwise
    """
    if (not token) or (token == "") or (len(token) <= 1):
        return False
    if ("html" in token ) or ("http" in token) or ("meta" in token):
        return False
    
    return True


def _lemmatize_tokens(tokens: List[str]) -> Iterable[str]:
    """
    Generator that yields valid lemmatized tokens
    """
    for token in tokens:
        lemma_token = re.sub('[0-9]', '', LEMMATIZER.lemmatize(token))
        if _good_token(lemma_token):
            yield lemma_token


def tokenize_text(text: str):
    """
    Tokenizes text based on RegexpTokenizer
    :param text: input sting
    :return: tokenized list of text
    """
    token_text = str(text).lower()
    token_text = re.sub('\[.*?\]', '', token_text)
    token_text = re.sub('https?://\S+|www\.\S+', '', token_text)
    token_text = re.sub('<.*?>+', '', token_text)
    token_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', token_text)
    token_text = re.sub('\n', '', token_text)
    token_text = re.sub('\w*\d\w*', '', token_text)
    no_contractions = contractions.fix(token_text)
    
    return TOKENIZER.tokenize(no_contractions)


def lemmatize_tokens(text: str):
    """
    Performs lemmatization using WordNetLemmatizer and conducts additional cleaning checks for tokenized text
    :param text: input string
    :return: lemmatized list of text
    """
    return list(_lemmatize_tokens(tokenize_text(text)))


def generate_lemma(texts: Union[List[str], str]):
    """
    Reduce the inflectional forms of each word in a text into a common base or root with lemmatization
    Enforces an is_ascii filter which means all tokens that are not strictly ascii will be cleaned out
    :param texts: <List [str] or str> List of texts, or single text
    :return: <Union[List[List[str], List[str]]>  list of lists of base words for each text or lists of texts
    """
    if isinstance(texts, list):
        clean_toks = [lemmatize_tokens(doc) for doc in texts]
        return clean_toks
    elif isinstance(texts, str):
        clean_toks = lemmatize_tokens(texts)
        return clean_toks
    else:
        TypeError(f"type: {type(texts)} is not supported use either list or str")


def hashtag_to_words(text: str) -> str:
    """
    Method to convert hashtags to words
    :param text: of a piece of text
    :return: text with hashtags converted to words
    """
    text = EMOJIS.sub(r"", text)
    text = HTML.sub(r"", text)

    for hashtag in HASHTAG.findall(text):
        text = text.replace(
            hashtag, 
            re.sub(r"([A-Z])", r" \1", hashtag)
        )

    return text.replace("#", "").strip()


def handle_urls(text: str) -> str:
    """
    Method to handle URLs
    :param text: a piece of text
    :return: text with URLs removed
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub('@[^\s]+', " ", text)
    text = text.replace("RT", "").replace("rt", "")

    return text.strip()


def remove_punct_noise(text: str) -> str:
    """
    Method to remove punctuation noise
    :param text: a piece of text
    :return: text with punctuation noise removed
    """
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text


def clean(text: str) -> str:
    """
    Clean text by removing URLs, HTML tags, line breaks, tabs,  and extra whitespace
    :param text: a piece of raw input text
    :return: prefixed clean text for embeddings
    """
    # raise error if text is empty or null
    if not text:
        raise NoText("Text is empty before cleaning")
    
    # convert hashtags to words
    text = hashtag_to_words(text)

    # remove line breaks and tabs
    for char in ["\n", "\t", "\\n", "\\t", "\r", "\\r"]:
        text = text.replace(char, " ")

    # remove URLs
    text = handle_urls(text)

    # raise error if text is empty after cleaning 
    if not text:
        raise NoText("Text is empty after cleaning")
    
    # normalize text with unicode
    text = unicodedata.normalize("NFKD", text).strip()
    text = html.unescape(text)

    # remove punctuation noise
    text = remove_punct_noise(text)

    # generate lemma
    text = " ".join(generate_lemma(text))

    # according to e5 documentation, text used for classificaiton tasks
    # require the "query: " prefix to use an embedding as a feature
    return f"query: {text}"
