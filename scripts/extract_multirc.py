import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from multibidaf.dataset_readers import MultiRCDatasetReader
from multibidaf.paths import Paths


def extract_multirc(input_file, output_file):
    reader = MultiRCDatasetReader(lazy=False)
    reader.read(Paths.DATA_ROOT / input_file)
    reader.write_to_tsv(Paths.DATA_ROOT / output_file)


if __name__ == "__main__":
    extract_multirc("multirc_train.json", "extracted_multirc_train.tsv")
    extract_multirc("multirc_dev.json", "extracted_multirc_dev.tsv")
