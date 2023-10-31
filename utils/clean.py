import re
import html
import unicodedata


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

    return text.strip()


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

    # according to e5 documentation, text used for classification tasks
    # require the "query: " prefix to use an embedding as a feature
    return f"query: {text}"
