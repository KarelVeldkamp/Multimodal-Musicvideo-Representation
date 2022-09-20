"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from thesis_project.mmnet.data import create_dataloader
from thesis_project.mmnet.train import train
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket('thesis_video_files')

def create_training_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            create_dataloader,
            inputs=['params:batch_size' ,'train_audio', 'train_video', 'params:train_size', 'params:opt'],
            outputs='train_dataloader'
        ),
        node(
            create_dataloader,
            inputs=['params:val_batch_size', 'val_audio', 'val_video', 'params:val_size', 'params:opt'],
            outputs='validation_dataloader'
        ),
        node(
            train,
            inputs=['params:opt', 'train_dataloader', 'validation_dataloader', 'params:val_batch_size'],
            outputs='model'
        )
    ])
