# -*- coding: utf-8 -*- noqa
"""
Created on Wed Nov 20 19:09:17 2024

@author: Joel Tapia Salvador
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc


def calculate_area_under_the_cruve(
        false_posotive_rate: np.darray,
        true_positive_rate: np.darray,
) -> float:
    """
    Calculate the area under the currve for a ROC curve.

    Parameters
    ----------
    false_posotive_rate : np.darray
        False positive rate for the ROC curve.
    true_positive_rate : np.darray
        True positive rate for the ROC curve.

    Returns
    -------
    float
        Value of the area under the curve.

    """
    return auc(false_posotive_rate, true_positive_rate)


def calculate_roc_curve_points(
        target_labels: np.ndarray,
        probabilities: np.ndarray,
        label_positive=None,
):
    """
    Calculate the points of the ROC curve.

    Parameters
    ----------
    target_labels : np.darray
        DESCRIPTION.
    probabilities : np.darray
        DESCRIPTION.

    Returns
    -------
    false_posotive_rate : numpy array
        DESCRIPTION.
    true_positive_rate : numpy array
        DESCRIPTION.
    thresholds : numpy array
        DESCRIPTION.

    """
    false_posotive_rate, true_positive_rate, thresholds = roc_curve(
        target_labels, probabilities, pos_label=label_positive)
    return false_posotive_rate, true_positive_rate, thresholds


def distance_to_perfection(x: float, y: float) -> float:
    """
    Calculate the distance from a threshold point to the point (0, 1).

    Parameters
    ----------
    x : float
        x point of the threshold (false_posotive_rate).
    y : float
        y point of the threhold (true_positive_rate).

    Returns
    -------
    float
        Distance.

    """
    return ((x) ** 2 + (1-y) ** 2) ** 0.5


def plot_roc(
        false_posotive_rate: np.darray,
        true_positive_rate: np.darray,
        area_under__the_curve: float,
        best_thr: None,
        title: "",
) -> None:
    """
    Plot the ROC curve with AUC and compare to random decission.

    Parameters
    ----------
    false_posotive_rate : numpy array
        DESCRIPTION.
    true_positive_rate : numpy array
        DESCRIPTION.
    area_under__the_curve : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Plot the ROC curve
    plt.figure()
    plt.plot(
        false_posotive_rate,
        true_positive_rate,
        label=f'ROC curve (area = {area_under__the_curve:0.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')

    if best_thr is not None:
        plt.plot(best_thr["point"], label="Best threshold: {value}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(title)

    plt.legend(loc="lower right")

    plt.savefig(title + ".png")

    plt.show()


def roc(
        target_labels: np.darray,
        probabilities: np.darray,
        label_positive=None,
        title=""
):
    """
    Do ROC analysis.

    Parameters
    ----------
    target_labels : np.darray
        DESCRIPTION.
    probabilities : np.darray
        DESCRIPTION.
    label_positive : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    fpr_array, tpr_array, thr_array = calculate_roc_curve_points(
        target_labels,
        probabilities,
        label_positive,
    )

    best_thr = {
        "value": None,
        "distance": float("inf"),
        "point": (None, None),
    }

    for point, thr in zip(zip(fpr_array, tpr_array), thr_array):
        distance = distance_to_perfection(point[0], point[1])
        if distance < best_thr["distance"]:
            best_thr = {
                "value": thr,
                "distance": distance,
                "point": point,
            }

    auc = calculate_area_under_the_cruve(fpr_array, tpr_array)

    plot_roc(fpr_array, tpr_array, auc, best_thr, title)
