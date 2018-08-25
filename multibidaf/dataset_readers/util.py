"""
Utilities for MultiRC dataset reader.
"""

import logging
from typing import Any, Dict, List, Tuple

from allennlp.data.fields import Field, TextField, MetadataField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def compute_sentence_start_list(tokenized_sentences: List[List[Token]]) -> List[int]:
    """
    Computes the start token index for each sentence in the paragraph.

    Parameters
    ----------
    tokenized_sentences : ``List[List[Token]]``
        An already-tokenized sentences which together constitute a paragraph.
    """
    sentence_start_list = [0]
    sentence_start = 0
    for i in range(1, len(tokenized_sentences)):
        sentence_start += len(tokenized_sentences[i - 1])
        sentence_start_list.append(sentence_start)
    return sentence_start_list


def make_reading_comprehension_instance_multirc(question_tokens: List[Token],
                                                passage_tokens: List[Token],
                                                token_indexers: Dict[str, TokenIndexer],
                                                passage_text: str,
                                                span_start_list: List[int],
                                                sentence_start_list: List[int],
                                                answer_texts: List[str],
                                                answer_labels: List[int],
                                                additional_metadata: Dict[str, Any] = None) -> Instance:
    """
    Converts a question, a passage, and an  answer (or answers) to an ``Instance`` for use
    in the Multi-BiDAF model.

    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual spans from the
        original passage that the model predicts as the sentences required to answer the question.
        This is used in official evaluation scripts.
    span_start_list : ``List[int]``
        The start token indices into ``passage_tokens`` to use as the sentences required to answer the
        question for training.  This is a list because (most of) the questions in the MultiRC dataset
        require reasoning over multiple sentences to be answered.
    sentence_start_list : ``List[int]``
        The start token indices of each sentence in the paragraph.
    answer_texts : ``List[str]``
        All answer option strings for the given question.  In MultiRC the number of correct
        answer-options for each question is not pre-specified and the correct answer(s) is not
        required to be a span in the text. This is put into the metadata for use with official
        evaluation scripts, but not used anywhere else.
    answer_labels : ``List[int]``
        The labels for each answer option - 1 for correct answer and and 0 otherwise. This is also put
        into the metadata for use with official evaluation scripts, but not used anywhere else.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, ``answer_texts``  and
        ``answer_labels`` keys. If you want any other metadata to be associated with each instance, you
        can pass that in here. This dictionary will get added to the ``metadata`` dictionary we already
        construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    fields['span_start'] = ListField([IndexField(span_start, passage_field) for span_start in span_start_list])

    # TODO: Consider convert `answer_texts` and `answer_labels` to `Field` type.
    metadata = {
            'original_passage': passage_text,
            'token_offsets': passage_offsets,
            'sentence_start_list': sentence_start_list,
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            'answer_texts': answer_texts,
            'answer_labels': answer_labels
    }

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)
