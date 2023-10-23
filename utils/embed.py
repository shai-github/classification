import torch.nn.functional as F
import numpy as np

from typing import Union
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


PRETRAINED_TF = 'intfloat/e5-small-v2'
TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_TF)
MODEL = AutoModel.from_pretrained(PRETRAINED_TF)

PRETRAINED_SBERT = 'sentence-transformers/all-MiniLM-L12-v2'
SBERT_EMBEDDER = SentenceTransformer(PRETRAINED_SBERT)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Average pooling of the last hidden states
    :param last_hidden_states: last hidden states
    :param attention_mask: attention mask
    :return: average pooled embeddings
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_tf(input_texts: list[str]) -> Tensor:
    """
    Generate embeddings from input texts using the pretrained model
    :param input_texts: list of input texts
    :return: normalized text embeddings
    """
    batch_dict = TOKENIZER(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = MODEL(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0]


def embed_sbert(input_texts: list[str]) -> np.ndarray:
    """
    Generates embedding using the SBERT model
    :param input_texts: list of input texts
    :return: text embeddings
    """
    embeddings = SBERT_EMBEDDER.encode(
        input_texts,
        show_progress_bar=False
    )

    return embeddings


def embed(input_text: list[str], use_sbert: bool = False) -> Union[Tensor, np.ndarray]:
    """
    Embeds input text with the pretrained model
    :param input_text: input text
    :param use_sbert: whether to use SBERT or TF
    :return: text embeddings
    """
    if use_sbert:
        return embed_sbert(input_text)
    else:
        return embed_tf(input_text)
    