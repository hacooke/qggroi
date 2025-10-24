import sqlite3
import pandas
import numpy as np

TOP_COL = 'top_det_pd'
BOT_COL = 'bot_det_pd'


def read_and_clean_data(db_file: str) -> pandas.DataFrame:
    df = read_from_database(db_file)
    df = format_columns(df)
    df = top_bottom_normalise(df)
    return df


def read_from_database(db_file: str) -> pandas.DataFrame:
    with sqlite3.connect(db_file, uri=False) as db:
        df = pandas.read_sql_query('select * from shot_data;', db)
    return df


def format_columns(df: pandas.DataFrame) -> pandas.DataFrame:
    df['timestamp'] = pandas.to_datetime(df['timestamp'])
    df['top'] = df[TOP_COL].apply(decode)
    df['bottom'] = df[BOT_COL].apply(decode)
    return df[['timestamp', 'top', 'bottom']]


def top_bottom_normalise(df: pandas.DataFrame) -> pandas.DataFrame:
    bottom_detector_total = np.sum(np.sum(df['bottom']))
    top_detector_total = np.sum(np.sum(df['top']))
    df['bottom'] = df['bottom'].apply(lambda x: x * (top_detector_total / bottom_detector_total))
    return df


def decode(data: bytes) -> np.ndarray:
    data_array = np.frombuffer(data, dtype=np.uint32)
    # enforce assumption that input data is a square 2D array
    size = len(data_array) ** .5
    assert int(size) == size
    size = int(size)
    data_array = np.reshape(data_array, (size, size))
    data_array = handle_overflow(data_array)
    return data_array


def handle_overflow(data: np.ndarray) -> np.ndarray:
    # return data
    new_data = data.astype('int')
    new_data[new_data > 3e9] -= np.iinfo(np.uint32).max
    return new_data
