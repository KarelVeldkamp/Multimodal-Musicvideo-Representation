"""
This is a boilerplate pipeline 'download_embeddings'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            download_embeddings,
            ['train_audio', 'train_video', 'params:train_set_size', 'params:download_batch_size', 'params:train'],
            None
        ),
        node(
            download_embeddings,
            ['validation_audio', 'validation_video', 'params:validation_set_size', 'params:download_batch_size', 'params:val'],
            None
        ),
        node(
            download_embeddings,
            ['test_audio', 'test_video', 'params:test_set_size', 'params:download_batch_size', 'params:test'],
            None
        )
    ])
