import torch.nn.functional as F
import numpy as np

from typing import Union
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings


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
