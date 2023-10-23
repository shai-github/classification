import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


PRETRAINED_MODEL = 'intfloat/e5-small-v2'
TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
MODEL = AutoModel.from_pretrained(PRETRAINED_MODEL)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Average pooling of the last hidden states
    :param last_hidden_states: last hidden states
    :param attention_mask: attention mask
    :return: average pooled embeddings
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_text(input_texts: list[str]) -> Tensor:
    """
    Generate batched embeddings from input texts using the pretrained model
    :param input_texts: list of input texts
    :return: batch normalized embeddings
    """
    batch_dict = TOKENIZER(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = MODEL(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0]
