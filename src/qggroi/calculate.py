import pandas
import numpy as np

from qggroi.roi import ROI


def decorate_data(df: pandas.DataFrame) -> tuple[pandas.DataFrame]:
    df['diff'] = df.apply(lambda row: row['top'] - row['bottom'], axis=1)
    image_columns = ['top', 'bottom', 'diff']
    df_stacked_images = df[image_columns].stack().groupby(level=1)
    df_stats = pandas.concat(
        {
            'mean': df_stacked_images.apply(lambda x: np.mean(np.array(x))),
            'std': df_stacked_images.apply(lambda x: np.std(np.array(x))),
        },
        axis=1
    )
    return df, df_stats


def integrate_rois(df: pandas.DataFrame, rois: list[ROI]) -> pandas.DataFrame:
    """Decorates dataframe with additional columns giving ROI integrals in each shot."""
    for detector in ['top', 'bottom']:
        roi_integrals = f'{detector}_roi_integrals'
        df[roi_integrals] = df[detector].apply(
            lambda x: [roi.integral(x) for roi in rois]
        )
        df[f'{detector}_total_integral'] = df[roi_integrals].apply(sum)
    return df
