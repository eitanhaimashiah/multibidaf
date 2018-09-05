import os
import sys
import json
import shutil
import numpy as np
import logging
from pprint import pprint, pformat

from allennlp.common import Params
from allennlp.models import load_archive
from allennlp.commands.fine_tune import fine_tune_model

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from multibidaf.dataset_readers import MultiRCDatasetReader
from multibidaf.models import MultipleBidirectionalAttentionFlow

# Log experiments to a log file and console.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('scripts/cross_validation.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


def grid_search(model_archive_path: str,
                config_file: str,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                step: float = 0.2):
    thresholds = []
    f1_m_scores = []
    for span_threshold in [0.25, 0.5, 0.75]:
        for true_threshold in [0.3, 0.5, 0.7]:
            for false_threshold in np.arange(0.3, true_threshold + step, step):
                thresholds.append((span_threshold, true_threshold, false_threshold))
                logger.info("#" * 100)
                logger.info("-" * 100)
                logger.info("The current setting is (span_threshold={}, true_threshold={}, false_threshold={})"
                            .format(span_threshold, true_threshold, false_threshold))
                logger.info("-" * 100)

                # Delete the serialization directory, if exists.
                if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
                    shutil.rmtree(serialization_dir, ignore_errors=True)

                # Train the MultiBiDAF model with the current settings.
                threshold_params = {
                                    "model": {
                                        "span_threshold": span_threshold,
                                        "true_threshold": true_threshold,
                                        "false_threshold": false_threshold
                                    }
                }
                archive = load_archive(model_archive_path,
                                       overrides=str(threshold_params))

                params = Params.from_file(config_file)
                fine_tune_model(archive.model,
                                params,
                                serialization_dir,
                                file_friendly_logging)

                # Add the best validation F1_m score of this run to f1_m_scores
                with open(os.path.join(serialization_dir, "metrics.json")) as f:
                    metrics = json.load(f)
                f1_m_scores.append(metrics["best_validation_f1_m"])
                logger.info(pformat(metrics))

    # Find the best setting
    max_f1_m = max(f1_m_scores)
    argmax = f1_m_scores.index(max(f1_m_scores))
    best_setting = thresholds[argmax]

    logger.info("*" * 100)
    logger.info("The best setting is (span_threshold={}, true_threshold={}, false_threshold={}) with F1_m={}"
                .format(best_setting[0], best_setting[1], best_setting[2], max_f1_m))
    logger.info("*" * 100)


if __name__ == "__main__":
    # grid_search("/tmp/squad_serialization/model.tar.gz",
    #             "training_config/multirc_config.json",
    #             "/tmp/multirc_serialization",
    #             step=0.2)

    grid_search("/tmp/test/squad_serialization/model.tar.gz",
                "multibidaf/tests/fixtures/multirc_experiment.json",
                "/tmp/test/multirc_serialization",
                step=0.2)

