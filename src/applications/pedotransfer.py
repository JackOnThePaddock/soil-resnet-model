"""Saxton & Rawls pedotransfer functions for soil hydraulic properties."""

import numpy as np


def saxton_rawls_wp(sand: float, clay: float, om: float) -> float:
    """
    Wilting point (1500 kPa) moisture content.

    Args:
        sand: Sand fraction (%)
        clay: Clay fraction (%)
        om: Organic matter (%)

    Returns:
        Wilting point volumetric water content (cm3/cm3)
    """
    wp_i = (
        -0.024 * sand / 100
        + 0.487 * clay / 100
        + 0.006 * om
        + 0.005 * (sand / 100) * om
        - 0.013 * (clay / 100) * om
        + 0.068 * (sand / 100) * (clay / 100)
        + 0.031
    )
    return wp_i + 0.14 * wp_i - 0.02


def saxton_rawls_fc(sand: float, clay: float, om: float) -> float:
    """
    Field capacity (33 kPa) moisture content.

    Args:
        sand: Sand fraction (%)
        clay: Clay fraction (%)
        om: Organic matter (%)

    Returns:
        Field capacity volumetric water content (cm3/cm3)
    """
    fc_i = (
        -0.251 * sand / 100
        + 0.195 * clay / 100
        + 0.011 * om
        + 0.006 * (sand / 100) * om
        - 0.027 * (clay / 100) * om
        + 0.452 * (sand / 100) * (clay / 100)
        + 0.299
    )
    return fc_i + 1.283 * fc_i * fc_i - 0.374 * fc_i - 0.015


def saxton_rawls_sat(sand: float, clay: float, om: float) -> float:
    """
    Saturation moisture content.

    Args:
        sand: Sand fraction (%)
        clay: Clay fraction (%)
        om: Organic matter (%)

    Returns:
        Saturation volumetric water content (cm3/cm3)
    """
    fc = saxton_rawls_fc(sand, clay, om)
    wp = saxton_rawls_wp(sand, clay, om)
    sat_i = (
        0.078
        + 0.278 * sand / 100
        + 0.034 * clay / 100
        + 0.022 * om
        - 0.018 * (sand / 100) * om
        - 0.027 * (clay / 100) * om
        - 0.584 * (sand / 100) * (clay / 100)
        + 0.078
    )
    return fc + sat_i - 0.097 * sand / 100 + 0.043


def plant_available_water(sand: float, clay: float, om: float, depth_cm: float = 100) -> float:
    """
    Plant Available Water Capacity (PAWC) in mm.

    Args:
        sand, clay, om: As above
        depth_cm: Soil depth in cm

    Returns:
        PAWC in mm
    """
    fc = saxton_rawls_fc(sand, clay, om)
    wp = saxton_rawls_wp(sand, clay, om)
    return max(0.0, (fc - wp) * depth_cm * 10)
