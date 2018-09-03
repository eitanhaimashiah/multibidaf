import json
import logging
import re
from typing import Dict, List, Tuple

from allennlp.common import Params
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from multibidaf.dataset_readers import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("multirc")
class MultiRCDatasetReader(DatasetReader):
    """
    Reads a JSON-formatted MultiRC file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``spans``, a
    ``ListField`` of ``IndexField`` representing start token indices of spans in ``passage`` required to
    answer the ``question``. We also add a ``MetadataField`` that stores the instance's ID, the question
    ID, the passage ID, the original passage text, the question tokens, the passage tokens, the gold
    answer strings, the gold answer labels, the token offsets into the original passage, and the start
    token indices of each sentence in the paragraph, accessible as ``metadata['qid']``,
    ``metadata['pid']``, ``metadata['original_passage']``, ``metadata['question_tokens']``,
    ``metadata['passage_tokens']``, ``metadata['answer_texts']``, ``metadata['answer_labels']``,
    ``metadata['token_offsets']``, and ``metadata['sentence_start_list']`` respectively. This is so that
    we can more easily use the official MultiRC evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the passage, the question and the answer options.
        See :class:`Tokenizer`. Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.
        See :class:`TokenIndexer`. Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        # Used by the ``write_to_tsv`` method.
        self._all_sentences = []
        self._all_answers = []

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            pid = article["id"]
            paragraph_json = article["paragraph"]

            # Remove the tags from the paragraph so that the text itself is left and
            # then tokenize it.
            original_paragraph = paragraph_json["text"]
            paragraph = re.sub(r"<b>Sent \d+: </b>", "", original_paragraph)\
                .replace("<br>", " ").strip()
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)

            # Split the paragraph to sentences, tokenize them and then compute the start
            # and end token indices for each sentence in the paragraph.
            sentences = re.split(r"<b>Sent \d+: </b>",
                                 original_paragraph.replace("<br>", ""))[1:]
            tokenized_sentences = self._tokenizer.batch_tokenize(sentences)
            sentence_starts = util.compute_sentence_starts(tokenized_sentences)

            self._all_sentences.extend(sentences)

            for question_answer in paragraph_json["questions"]:
                question_text = question_answer["question"].strip().replace("\n", "")
                answer_texts = [answer["text"] for answer in question_answer["answers"]]
                answer_labels = [int(answer["isAnswer"]) for answer in question_answer["answers"]]
                used_sentences_nums = question_answer["sentences_used"]
                span_starts = [sentence_starts[sentence_num-1]
                               for sentence_num in used_sentences_nums]
                qid = question_answer["idx"]

                instance = self.text_to_instance(question_text,
                                                 paragraph,
                                                 span_starts,
                                                 sentence_starts,
                                                 answer_texts,
                                                 answer_labels,
                                                 pid,
                                                 qid,
                                                 tokenized_paragraph)

                self._all_answers.extend(answer_texts)

                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         span_starts: List[int] = None,
                         sentence_starts: List[int] = None,
                         answer_texts: List[str] = None,
                         answer_labels: List[int] = None,
                         pid: str = None,
                         qid: str = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)

        return util.make_reading_comprehension_instance_multirc(self._tokenizer.tokenize(question_text),
                                                                passage_tokens,
                                                                self._token_indexers,
                                                                passage_text,
                                                                span_starts,
                                                                sentence_starts,
                                                                answer_texts,
                                                                answer_labels,
                                                                {'pid': pid, 'qid': qid})

    def get_documents(self):
        """
        Returns the paragraph sentences and answers from all the examples
        in the database.
        """
        return list(set().union(self._all_sentences, self._all_answers))

