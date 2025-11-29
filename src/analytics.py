from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st


def compute_ndvi_cube(dataset: xr.Dataset) -> xr.DataArray:
    """
    NDVI = (NIR - RED) / (NIR + RED) with NIR=B08, RED=B04.
    """
    if "B08" not in dataset or "B04" not in dataset:
        raise ValueError("Dataset must contain B08 and B04 for NDVI.")

    nir = dataset["B08"]
    red = dataset["B04"]
    ndvi = (nir - red) / (nir + red)
    return ndvi


def summarize_ndvi_per_frame(ndvi: xr.DataArray) -> None:
    """
    Computes and plots the mean NDVI per time slice as a simple time series.
    """
    st.subheader("NDVI summary")
    st.caption("Average NDVI per acquisition – basic vegetation dynamics over time.")

    # Mean over spatial dimensions (assume y/x naming)
    spatial_dims = [d for d in ndvi.dims if d not in ("time",)]
    series = ndvi.mean(dim=spatial_dims, skipna=True)

    # Convert to numpy for plotting
    times = series["time"].values
    values = series.values

    fig, ax = plt.subplots()
    ax.plot(times, values, marker="o")
    ax.set_ylabel("Mean NDVI")
    ax.set_xlabel("Acquisition date")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    st.pyplot(fig)
