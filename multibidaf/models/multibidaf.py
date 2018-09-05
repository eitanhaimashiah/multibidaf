import logging
from typing import Any, Dict, List, Optional

import torch
import random
from overrides import overrides
from sklearn.externals import joblib

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.models import BidirectionalAttentionFlow, Model

from multibidaf.models.util import sentence_start_mask
from multibidaf.training.functional import multi_nll_loss
from multibidaf.training.metrics import SpanStartMetrics
from multibidaf.training.metrics import MultiRCMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("multibidaf")
class MultipleBidirectionalAttentionFlow(BidirectionalAttentionFlow):
    """
    This class implements our `Multiple Sentences Bidirectional Attention Flow model
    <https://github.com/eitanhaimashiah/multibidaf.git>`_
    which extends the `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    of Seo et al. (2017) so that it can be used for the `Multi-RC dataset <http://cogcomp.org/multirc/>`_.

    The adjustment we offer requires a modification to the output layer of the BiDAF network only.
    The original output layer in BiDAF is intended for the QA task, in which the model is required
    to find a sub-phrase of a paragraph to answer a query. The phrase is derived by predicting the
    start and the end indices of the phrase in the paragraph. However, the MultiRC dataset
    suggests a reading comprehension challenge in which answering each question requires reasoning
    over multiple sentences. Therefore, instead of the model outputting only a single phrase, it
    will output up to four phrases (the number of sentences required to answer each question in
    MultiRC is between 2 and 4). In addition, we notice that each such phrase in MultiRC's
    paragraph may only be a sentence, and thus we only need to predict the start indices; the
    predictions for the end indices are omitted from the loss function.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start.
    span_end_encoder : ``Seq2SeqEncoder``, optional.
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end. Only relevant when working on SQuAD.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    tfidf_path: ``str``, optional (default=``str``)
        A path to the trained TfidfVectorizer model, used for document similarity.
    span_threshold: ``float``, optional (default=0.5)
        A threshold to determine how many spans (up to 4) will be outputted.
    true_threshold: ``float``, optional (default=0.7)
        If ``true_threshold`` <= the maximum similarity of an answer-option and predicted span,
        then that answer-option will be predicted as a correct answer.
    false_threshold: ``float``, optional (default=0.3)
        If ``false_threshold`` >= the maximum similarity of an answer-option and predicted span,
        then that answer-option will be predicted as an incorrect answer.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder = None,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 tfidf_path: str = None,
                 span_threshold: float = 0.6,
                 true_threshold: float = 0.1,
                 false_threshold: float = 0.05) -> None:

        self._is_squad = False
        span_end_encoder = span_end_encoder or Seq2SeqEncoder.from_params(Params({"type": "lstm",
                                                                                  "input_size": 70,
                                                                                  "hidden_size": 10,
                                                                                  "num_layers": 1}))
        self._span_start_metrics = SpanStartMetrics()
        self._multirc_metrics = MultiRCMetrics()
        self._tfidf_vec = joblib.load(tfidf_path)
        self._span_threshold = span_threshold
        self._true_threshold = true_threshold
        self._false_threshold = false_threshold
        logger.info("-" * 100)
        logger.info("The current setting is (span_threshold={}, true_threshold={}, false_threshold={})"
                    .format(self._span_threshold, self._true_threshold, self._false_threshold))
        logger.info("-" * 100)

        super(MultipleBidirectionalAttentionFlow, self).__init__(vocab,
                                                                 text_field_embedder,
                                                                 num_highway_layers,
                                                                 phrase_layer,
                                                                 similarity_function,
                                                                 modeling_layer,
                                                                 span_end_encoder,
                                                                 dropout,
                                                                 mask_lstms,
                                                                 initializer,
                                                                 regularizer)


    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
            Whereas ``span_start`` in SQuAD contains only one value per example in the batch and
            may be every token in the passage, ``span_start`` in MultiRC contains up to four values
            which must be a beginning of a sentence in the passage.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
            Only relevant when working on SQuAD.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key. ``metadata`` in MultiRC
            contains the same fields but in addition it also contains the passage ID, the labels for
            each answer option, and the start token indices of each sentence in the paragraph, for
            each instance in the batch, accessible as ``pid``, ``answer_labels`` and
            ``sentence_starts``.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive). Only relevant when working on SQuAD.
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``. Only relevant when working on SQuAD.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span. Shape is ``(batch_size, 2)``
            and each offset is a token index. Only relevant when working on SQuAD.
        best_span_starts : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` to find the most
            probable span starts. Shape is ``(batch_size, 4)`` and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question. Only relevant when working on SQuAD.
        """
        # If this is a SQuAD instance, call the parent's class method.
        self._is_squad = 'sentence_starts' not in metadata[0]
        if self._is_squad:
            return super(MultipleBidirectionalAttentionFlow, self).forward(question,
                                                                           passage,
                                                                           span_start,
                                                                           span_end,
                                                                           metadata)

        # Otherwise, handle a MultiRC instance.
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)

        # Reset the logits corresponding to non-start indices.
        sent_start_mask = sentence_start_mask(metadata, passage_length)
        mask = passage_mask * sent_start_mask
        span_start_logits = util.replace_masked_values(span_start_logits, mask, -1e7)
        span_start_probs = util.masked_softmax(span_start_logits, mask)
        best_span_starts = self.get_best_span_starts(span_start_probs)

        output_dict = {
                "passage_question_attention": passage_question_attention,
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "best_span_starts": best_span_starts
                }

        # Compute the loss for training.
        if span_start is not None:
            loss = multi_nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            output_dict["loss"] = loss

        # Compute the EM, F1_m, F1_a and accuracy on MultiRC and add the tokenized input to the output.
        if metadata is not None:
            # TODO: Check exactly what's needed for the prediction phase
            pids = []
            qids = []
            question_tokens = []
            passage_tokens = []
            answer_texts = []
            answer_labels = []
            scores = []
            for i in range(batch_size):
                pids.append(metadata[i]['pid'])
                qids.append(metadata[i]['qid'])
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                _answer_texts = metadata[i]['answer_texts']
                answer_texts.append(_answer_texts)

                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                sentence_starts = metadata[i]['sentence_starts']
                predicted_span_starts = best_span_starts[i].detach().cpu().numpy()

                predicted_span_strings = []
                for j in range(4):
                    # TODO: Make sure that's the logic
                    predicted_span_start = predicted_span_starts[j]
                    if predicted_span_start == -1:
                        continue

                    start_offset = offsets[predicted_span_start][0]
                    predicted_next_span_start = sentence_starts.index(predicted_span_start) + 1
                    if predicted_next_span_start < len(sentence_starts) - 1:
                        predicted_span_end = sentence_starts[predicted_next_span_start] - 1
                        end_offset = offsets[predicted_span_end][1]
                        predicted_span_string = passage_str[start_offset:end_offset]
                    else:
                        predicted_span_string = passage_str[start_offset:]
                    predicted_span_strings.append(predicted_span_string.lower())

                _scores = self.get_scores(predicted_span_strings, _answer_texts)
                scores.append(_scores)
                _answer_labels = metadata[i].get('answer_labels', [])
                if _answer_labels:
                    answer_labels.append(_answer_labels)
                    self._multirc_metrics(_scores, _answer_labels)

            output_dict['pid'] = pids
            output_dict['qid'] = qids
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['answer_texts'] = answer_texts
            output_dict['scores'] = scores
            if answer_labels:
                output_dict['answer_labels'] = answer_labels
                self._span_start_metrics(best_span_starts, span_start.squeeze(-1))

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # If this is a SQuAD instance, call the parent's class method.
        if self._is_squad:
            return super(MultipleBidirectionalAttentionFlow, self).get_metrics(reset)

        # Otherwise, handle a MultiRC instance.
        ss_exact_match, ss_accuracy, ss_f1_m_score, ss_f1_a_score = self._span_start_metrics.get_metric(reset)
        exact_match, accuracy, f1_m_score, f1_a_score = self._multirc_metrics.get_metric(reset)

        return {
                'ss_accuracy': ss_accuracy,
                'ss_em': ss_exact_match,
                'ss_f1_a': ss_f1_a_score,
                'ss_f1_m': ss_f1_m_score,
                'accuracy': accuracy,
                'em': exact_match,
                'f1_a': f1_a_score,
                'f1_m': f1_m_score
                }

    def get_best_span_starts(self, span_start_probs: torch.Tensor) -> torch.Tensor:
        if span_start_probs.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")

        batch_size, passage_length = span_start_probs.size()
        top_start_span_probs, best_span_starts = span_start_probs.topk(4, -1)
        for b in range(batch_size):  # pylint: disable=invalid-name
            j = 1
            span_prob_sum = top_start_span_probs[b, 0]
            while span_prob_sum < self._span_threshold and j < 4:
                span_prob_sum += top_start_span_probs[b, j]
                j += 1
            best_span_starts[b, j:] = -1

        return best_span_starts

    def get_scores(self, predicted_span_strings: List[str],
                   answers: List[str]) -> List[int]:
        """
        Returns the score of each answer-option, 1 for a predicted correct answer and 0 otherwise.
        The score is computed by the maximum similarity between an answer and each predicted span
        string.
        """
        scores = []
        max_similarities = []
        for answer in answers:
            max_similarity = self._max_cosine_similarity(predicted_span_strings, answer)
            max_similarities.append(max_similarity)
            if max_similarity >= self._true_threshold:
                score = 1
            elif max_similarity <= self._false_threshold:
                score = 0
            else:
                score = random.choice([0, 1])

            scores.append(score)

        if sum(scores) == 0:
            argmax = max_similarities.index(max(max_similarities))
            scores[argmax] = 1
        return scores

    def _max_cosine_similarity(self, xs: List[str],
                               y: str) -> float:
        """
        Returns the maximum cosine similarity score of y and each element in xs.
        """
        tfidf_matrix = self._tfidf_vec.transform([y] + xs)
        cosine_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        return cosine_matrix[0, 1:].max()

    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["pid", "qid", "scores"]
        unwanted_keys = set(output_dict.keys()) - set(keys)
        for unwanted_key in unwanted_keys:
            del output_dict[unwanted_key]
        return output_dict


