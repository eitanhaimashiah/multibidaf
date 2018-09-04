import os
import sys
import json
import shutil
import numpy as np

from allennlp.common import Params
from allennlp.models import load_archive
from allennlp.commands.fine_tune import fine_tune_model

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from multibidaf.dataset_readers import MultiRCDatasetReader
from multibidaf.models import MultipleBidirectionalAttentionFlow


def grid_search(model_archive_path: str,
                config_file: str,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                step: float = 0.2):
    thresholds = []
    f1_m_scores = []
    for span_threshold in np.arange(0, 1, step):
        for true_threshold in np.arange(0, 1, step):
            for false_threshold in np.arange(0, true_threshold + step, step):
                thresholds.append((span_threshold, true_threshold, false_threshold))
                print("#" * 100)
                print("The current setting is (span_threshold={}, true_threshold={}, false_threshold={})"
                      .format(span_threshold, true_threshold, false_threshold))
                print("#" * 100)

                # Delete the serialization directory, if exists.
                if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
                    shutil.rmtree(serialization_dir, ignore_errors=True)

                # Train the MultiBiDAF model with the current settings.
                threshold_params = {"span_threshold": span_threshold,
                                    "true_threshold": true_threshold,
                                    "false_threshold": false_threshold}
                archive = load_archive(model_archive_path,
                                       overrides=str(threshold_params))
                params = Params.from_file(config_file)
                params.params.update()
                fine_tune_model(archive.model,
                                params,
                                serialization_dir,
                                file_friendly_logging)

                # Add the best validation F1_m score of this run to f1_m_scores
                with open(os.path.join(serialization_dir, "metrics.json")) as f:
                    metrics = json.load(f)
                f1_m_scores.append(metrics["best_validation_f1_m"])

    # Find the best setting
    max_f1_m = max(f1_m_scores)
    argmax = f1_m_scores.index(max(f1_m_scores))
    best_setting = thresholds[argmax]

    print("#" * 100)
    print("The best setting is (span_threshold={}, true_threshold={}, false_threshold={}) with F1_m={}"
          .format(best_setting[0], best_setting[1], best_setting[2], max_f1_m))
    print("#" * 100)


if __name__ == "__main__":
    grid_search("/tmp/squad_serialization/model.tar.gz",
                "training_config/multirc_config.json",
                "/tmp/multirc_serialization",
                step=0.2)

    # grid_search("/tmp/test/squad_serialization/model.tar.gz",
    #             "multibidaf/tests/fixtures/multirc_experiment.json",
    #             "/tmp/test/multirc_serialization",
    #             step=0.2)

