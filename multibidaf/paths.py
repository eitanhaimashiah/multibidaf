import pathlib


class Paths:
    PROJECT_ROOT = (pathlib.Path(__file__).parent / "..").resolve()  # pylint: disable=no-member
    DATA_ROOT = PROJECT_ROOT / "data"
    TRAINED_MODELS_ROOT = PROJECT_ROOT / "trained_models"
    MODULE_ROOT = PROJECT_ROOT / "multibidaf"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"


