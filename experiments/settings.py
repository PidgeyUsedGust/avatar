from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor

from avatar.evaluate import Game


class Experiment:
    """Some global settings for experiments."""

    samples = 1000
    games = 400
    rounds = 10

    # selection
    select = 16

    @staticmethod
    def get_estimator(task: str):
        if task == "classification":
            return XGBClassifier(use_label_encoder=False, verbosity=0)
        return XGBRegressor(use_label_encoder=False, verbosity=0)

    @staticmethod
    def get_evaluator(task: str) -> Game:
        return Game(
            estimator=Experiment.get_estimator(task),
            rounds=Experiment.rounds,
            samples=2e16,
        )

    @staticmethod
    def get_file(experiment: Path, name: str) -> Path:
        out_file = (
            experiment.parent.parent.parent
            / "results"
            / experiment.parent.name
            / "{}.json".format(name)
        )
        out_file.parent.mkdir(exist_ok=True, parents=True)
        return out_file
