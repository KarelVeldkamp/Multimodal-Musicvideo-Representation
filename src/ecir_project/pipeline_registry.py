"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from thesis_project.pipelines.train_model.pipeline import create_training_pipeline
from thesis_project.pipelines.download_embeddings.pipeline import create_pipeline as download
from thesis_project.pipelines.genre_classification.pipeline import create_pipeline as genre_classification


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": create_training_pipeline(),
            "training": create_training_pipeline(),
            "download": download(),
            'genre': genre_classification()}
