import sys

from qggroi import data as qgg_data
from qggroi import calculate as qgg_calculate
from qggroi import locate as qgg_locate
from qggroi import plot as qgg_plot
from qggroi import output as qgg_output


def main():
    # Core algorithm
    cleaned_data = qgg_data.read_and_clean_data(sys.argv[1])
    decorated_data, per_pixel_stats = qgg_calculate.decorate_data(cleaned_data)
    rois = qgg_locate.locate_rois(per_pixel_stats['std']['diff'], limit=8)
    roi_summary = qgg_output.summarise_rois(rois, df_stat=per_pixel_stats)
    roistat_data = qgg_calculate.integrate_rois(decorated_data, rois)
    print(roi_summary)

    # Plots for presentation
    std_image = per_pixel_stats['std']['diff']

    # Total integrals as function of time
    fig = qgg_plot.plot_total_integrals(roistat_data)
    qgg_plot.save_figure(fig, 'roi_totals.pdf')

    # Lissajous plots per ROI
    figs = qgg_plot.plot_per_roi_lissajous(roistat_data)
    for i, fig in enumerate(figs):
        qgg_plot.save_figure(fig, f'roi_{i}_lissajous.pdf')

    # Integrals per ROI
    figs = qgg_plot.plot_per_roi_integrals(roistat_data)
    for i, fig in enumerate(figs):
        qgg_plot.save_figure(fig, f'roi_{i}_integrals.pdf')

    # ROI locations
    for i, roi in enumerate(rois):
        fig = qgg_plot.plot_roi(roi, std_image)
        qgg_plot.save_figure(fig, f'roi_{i}.pdf')

    # Example profiles for peak finding
    roi = rois[5]
    fig = qgg_plot.plot_step(
        *qgg_plot.profile(std_image, 'y')[::-1],
        xlabel='$y$',
        color='black'
    )
    qgg_plot.save_figure(fig, 'example_yprofile.pdf')
    fig = qgg_plot.plot_step(
        *qgg_plot.profile(std_image, 'x', bounds=roi.y_bounds),
        xlabel=f'$x$ (for ${roi.y_bounds[0]} \\leq y \\leq {roi.y_bounds[1]}$)',
        color='black'
    )
    qgg_plot.save_figure(fig, 'example_xprofile.pdf')

    fig = qgg_plot.plot_image_and_profiles(std_image)
    qgg_plot.save_figure(fig, 'std_diff.pdf')


if __name__ == "__main__":
    main()
