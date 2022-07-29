"""
This is a boilerplate pipeline 'genre_classification'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            create_dataloaders,
            inputs=['train_audio', 'val_audio', 'test_audio', 'params:opt_genre', 'metadata'],
            outputs=['train_dataloader', 'val_dataloader', 'test_dataloader']
        ),
        node(
            create_model,
            inputs=['params:input_size', 'params:opt'],
            outputs='network'
        ),
        node(
            train_model,
            inputs=['network', 'train_dataloader', 'val_dataloader', 'params:opt_genre', 'params:modality'],
            outputs='trained_network'
        ),
        node(
            evaluate,
            inputs=['trained_network', 'test_dataloader', 'params:opt_genre'],
            outputs='results'
        )
    ])
