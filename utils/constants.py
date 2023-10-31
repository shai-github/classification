from nltk.corpus import stopwords


TWITTER_SPECIFIC = [
    "rt", "yoooooouuuu", "yoooo", "yooo", "loool", "lmfaoooo", "yoooooooooooooo", "lmfaoooooooo", "squaaaaad", "yoooo", "yoooou",
    "ah", ""
]

TOTAL_WORDS = list(set(stopwords.words('english') + TWITTER_SPECIFIC))
FOR_REMOVAL = [word for word in TOTAL_WORDS if word != "not"]
