from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('multiple-sentences-reading-comprehension')
class MultiBidafPredictor(Predictor):
    """
    Predictor for the :class:`~multibidaf.models.multibidaf.MultiBidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a reading comprehension prediction on the supplied input.
        See http://cogcomp.org/multirc/ for more information about the task of reading comprehension over
        multiple sentences.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        raise NotImplementedError

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        raise NotImplementedError
