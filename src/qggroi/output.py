import pandas

from qggroi.roi import ROI


def summarise_rois(rois: list[ROI], **kwargs) -> str:
    rtn = ''
    delim = f'{"":=<80}\n'
    for i, roi in enumerate(rois):
        rtn += delim
        roi_title = f' ROI {i+1} '
        rtn += f'---{roi_title:-<77}\n'
        rtn += roi_summary(roi, **kwargs) + '\n'
    rtn += delim
    return rtn


def roi_summary(roi: ROI, df_stat: pandas.DataFrame) -> str:
    return (
        f"{roi.x_bounds[0]} <= x <= {roi.x_bounds[1]}\n"
        f"{roi.y_bounds[0]} <= y <= {roi.y_bounds[1]}\n"
        f"Standard deviation integral: {roi.integral(df_stat['std']['diff']):.2e}\n"
        f"Mean integral (top): {roi.integral(df_stat['mean']['top']):.2e}\n"
        f"Mean integral (bottom): {roi.integral(df_stat['mean']['bottom']):.2e}\n"
    )


def summarise_rois_tex(rois: list[ROI], **kwargs) -> str:
    rtn = ''
    for i, roi in enumerate(rois):
        rtn += roi_summary_tex(i, roi, **kwargs) + '\n'
    return rtn


def roi_summary_tex(i: int, roi: ROI, df_stat: pandas.DataFrame) -> str:
    """Quick method to output results to beamer slides."""
    return (r'''\begin{frame}
    \frametitle{ROI '''f'{i+1}'r'''}
\begin{minipage}{.5\textwidth}
\includegraphics[width=\textwidth]{resources/roi_'''f'{i}'r'''.pdf}
\end{minipage}\hfill
\begin{minipage}{.5\textwidth}
\includegraphics[width=\textwidth]{resources/roi_'''f'{i}'r'''_integrals.pdf}
\end{minipage}
\begin{table}
\centering
\begin{tabular}{l c}
  $x$-bounds & $'''f'{roi.x_bounds[0]}'r''' \leq x \leq '''f'{roi.x_bounds[1]}'r'''$ \\
  $y$-bounds & $'''f'{roi.y_bounds[0]}'r''' \leq y \leq '''f'{roi.y_bounds[1]}'r'''$ \\
  Standard deviation integral & $'''f'{roi.integral(df_stat["std"]["diff"]):.2e}'r'''$ \\
  Mean integral (top) & $'''f'{roi.integral(df_stat["mean"]["top"]):.2e}'r'''$ \\
  Mean integral (bottom) & $'''f'{roi.integral(df_stat["mean"]["bottom"]):.2e}'r'''$ \\
\end{tabular}
\end{table}
\end{frame}''')
