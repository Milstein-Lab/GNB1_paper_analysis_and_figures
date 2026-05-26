#!/bin/bash

# Check if a figure number was provided
if [ -z "$1" ]; then
    echo "Usage: ./plot_single.sh <figure_number>"
    echo "Example: ./plot_single.sh 1"
    echo "Example: ./plot_single.sh S1"
    exit 1
fi

FIG=$1

# Map input to function names
case $FIG in
    1)
        FUNC="plot_figure_1_behavior"
        ;;
    2)
        FUNC="plot_figure_2_physiology"
        ;;
    3)
        FUNC="plot_figure_3_morphology"
        ;;
    4)
        FUNC="plot_figure_4_Unitary_E_I_Breakdown"
        ;;
    5)
        FUNC="plot_figure_5_EI_frequency_dependence"
        ;;
    6)
        FUNC="plot_figure_6_Supralinear_E_I"
        ;;
    7)
        FUNC="plot_figure_7_dendritic"
        ;;
    8)
        FUNC="plot_figure_8_GIRK"
        ;;
    S1|1s|s1)
        FUNC="plot_supplemental_figure_1"
        ;;
    S2|2s|s2)
        FUNC="plot_supplemental_figure_2"
        ;;
    S3|3s|s3)
        FUNC="plot_supplemental_figure_3"
        ;;
    *)
        echo "Error: Unknown figure '$FIG'"
        echo "Supported: 1, 2, 3, 4, 5, 6, 7, 8, S1, S2, S3"
        exit 1
        ;;
esac

echo "--- Generating Figure $FIG ($FUNC) ---"
python3 -c "from generate_figures import setup_publication_style, $FUNC; setup_publication_style(); $FUNC()"
