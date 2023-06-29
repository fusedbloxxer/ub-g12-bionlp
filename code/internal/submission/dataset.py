from datasets import ClassLabel, Dataset, Features, NamedSplit, Value
from datasets import DatasetDict
import pathlib as pb
import pandas as pd
import typing as t


def read_mediqa_subset(path: pb.Path, features: Features, split: str) -> Dataset:
    dataframe: pd.DataFrame = pd.read_csv(path, index_col=0).rename(columns={
        'section_header': 'header', 'section_text': 'summary',
    })
    dataframe.index.names = ['id']

    labels: pd.Series = dataframe['header'].apply(features['label'].str2int)
    dataframe.insert(len(dataframe.columns), column='label', value=labels)

    index: pd.Series = dataframe.index.to_series().apply(lambda x: x if isinstance(x, int) or x[0] != 'T' else x.split(' ')[1])
    dataframe.set_index(keys=index, inplace=True)
    dataset: Dataset = Dataset.from_pandas(dataframe, features, split=NamedSplit(split))
    return dataset


def read_mediqa_dataset(path: pb.Path) -> DatasetDict:
    # Predefine the expected paths to the data
    augmented_path: pb.Path = path / 'augmented' / 'MTS-Dialog-AugmentedSet-FR-ES.csv'
    test_path: pb.Path = path / 'test' / 'MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv'
    valid_path: pb.Path = path / 'valid' / 'MTS-Dialog-ValidationSet.csv'
    train_path: pb.Path = path / 'train' / 'MTS-Dialog-TrainingSet.csv'

    # Fetch all unique section headers
    section_headers = list(pd.read_csv(augmented_path, index_col=0)['section_header'].unique())

    # Define the types
    features = Features({
        'id': Value(dtype='uint32'),
        'header': Value(dtype='string'),
        'summary': Value(dtype='string'),
        'dialogue': Value(dtype='string'),
        'label': ClassLabel(names=section_headers)
    })

    # Read all datasets
    augmented_dataset: Dataset = read_mediqa_subset(augmented_path, features, split='augmented')
    valid_dataset: Dataset = read_mediqa_subset(valid_path, features, split='valid')
    train_dataset: Dataset = read_mediqa_subset(train_path, features, split='train')
    test_dataset: Dataset = read_mediqa_subset(test_path, features, split='test')

    # Aggregate them for ease of use
    return DatasetDict(**{
        'augmented': augmented_dataset,
        'valid': valid_dataset,
        'train': train_dataset,
        'test': test_dataset,
    })

