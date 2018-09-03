# pylint: disable=no-self-use
import pytest

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from multibidaf.dataset_readers import MultiRCDatasetReader
from multibidaf.paths import Paths


class TestMultiRCDatasetReader(AllenNlpTestCase):
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy=False):
        reader = MultiRCDatasetReader(lazy=lazy)
        instances = ensure_list(reader.read(Paths.FIXTURES_ROOT / "multirc.json"))

        # The start and end indices of the first sentences in each paragraph
        sentence_start_lists = [
            [0, 6, 28, 76, 92, 97],
            [0, 28, 64, 90, 129, 136, 148, 167, 177, 212, 244, 260, 317]
        ]

        expected_instances = [
            {
                "passage_start": ["Animated", "history", "of"],
                "passage_end": ["numbers", ")", "."],
                "question": ["Does", "the", "author"],
                "span_start": [sentence_start_lists[0][0], sentence_start_lists[0][1],
                               sentence_start_lists[0][2], sentence_start_lists[0][4]],
                "sentence_start_list": sentence_start_lists[0],
                "first_answer_text": "Yes",
                "last_answer_text": "No",
                "answer_labels": [1, 1, 0]
            },
            {
                "passage_start": ["Animated", "history", "of"],
                "passage_end": ["numbers", ")", "."],
                "question": ["Which", "key", "message(s"],
                "span_start": [sentence_start_lists[0][1], sentence_start_lists[0][5]],
                "sentence_start_list": sentence_start_lists[0],
                "first_answer_text": "The strategy to promote \"gun rights\" for white "
                                     "people while outlawing it for black people allowed "
                                     "racisim to continue without allowing to KKK to flourish",
                "last_answer_text": "The strategy to promote the KKK",
                "answer_labels": [1, 0, 0, 0]
            },
            {
                "passage_start": ["Before", "the", "mysterious"],
                "passage_end": ["wasted", ".", "\""],
                "question": ["What", "are", "two"],
                "span_start": [sentence_start_lists[1][1], sentence_start_lists[1][2],
                               sentence_start_lists[1][3]],
                "sentence_start_list": sentence_start_lists[1][:6],
                "first_answer_text": "Tornadoes",
                "last_answer_text": "Tsunamis",
                "answer_labels": [0, 0, 1, 1, 0, 1]
            },
            {
                "passage_start": ["Before", "the", "mysterious"],
                "passage_end": ["wasted", ".", "\""],
                "question": ["Why", "are", "Chinese"],
                "span_start": [sentence_start_lists[1][8], sentence_start_lists[1][9],
                               sentence_start_lists[1][10], sentence_start_lists[1][12]],
                "sentence_start_list": sentence_start_lists[1][:6],
                "first_answer_text": "Because Malaysia stated that the plane may have been flown into the Indian Ocean",
                "last_answer_text": "Because many passengers aboard the aircraft were from Chine and Vietnam",
                "answer_labels": [1, 1, 1, 1, 0, 0, 0, 0, 1]
            },
            {
                "passage_start": ["Before", "the", "mysterious"],
                "passage_end": ["wasted", ".", "\""],
                "question": ["What", "has", "the"],
                "span_start": [sentence_start_lists[1][11], sentence_start_lists[1][12]],
                "sentence_start_list": sentence_start_lists[1][:6],
                "first_answer_text": "The time for finding the terrorists involved has been wasted",
                "last_answer_text": "Locating survivors",
                "answer_labels": [0, 1, 0, 0, 0, 1, 0, 1, 1]
            }
        ]

        assert len(instances) == 5
        for instance, expected_instance in zip(instances, expected_instances):
            self._assert_instance(instance, expected_instance)

    @staticmethod
    def _assert_instance(instance, expected_instance):
        fields = instance.fields
        metadata = fields["metadata"].metadata

        assert [t.text for t in fields["passage"].tokens[:3]] == expected_instance["passage_start"]
        assert [t.text for t in fields["passage"].tokens[-3:]] == expected_instance["passage_end"]
        assert [t.text for t in fields["question"].tokens[:3]] == expected_instance["question"]
        assert [s.sequence_index for s in fields["span_start"].field_list] == expected_instance["span_start"]
        assert metadata["sentence_start_list"][:6] == expected_instance["sentence_start_list"]
        assert metadata["answer_texts"][0] == expected_instance["first_answer_text"]
        assert metadata["answer_texts"][-1] == expected_instance["last_answer_text"]
        assert metadata["answer_labels"] == expected_instance["answer_labels"]

    def test_can_build_from_params(self):
        reader = MultiRCDatasetReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'
