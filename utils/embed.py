import numpy as np

from sentence_transformers import SentenceTransformer


PRETRAINED_TF = SentenceTransformer('intfloat/e5-large-v2')


def embed(input_text: str) -> np.ndarray:
    """
    Generates embedding using sentence transformers
    :param input_text: string of input text
    :return: text embeddings
    """
    embeddings = PRETRAINED_TF.encode(
        input_text,
        show_progress_bar=False
    )

    return embeddings
