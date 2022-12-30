"""Rechung Zarr archive into 100MB chunks."""

# Third-party
import numpy as np
import xarray as xr

# COMBINE ALL DATA

# data_2022 = xr.open_zarr("/scratch/sadamov/pyprojects_data/aldernet/data2022")
# data_2021 = xr.open_zarr("/scratch/sadamov/pyprojects_data/aldernet/data2021")
# data_2020 = xr.open_zarr("/scratch/sadamov/pyprojects_data/aldernet/data2020")

# data_2020.to_zarr("/scratch/sadamov/pyprojects_data/aldernet/data")
# data_2021.to_zarr("/scratch/sadamov/pyprojects_data/aldernet/data",
#    mode="a", append_dim="valid_time")
# data_2022.to_zarr("/scratch/sadamov/pyprojects_data/aldernet/data",
#    mode="a", append_dim="valid_time")

# After this I had to manually remove the data variable "nominalTop"

data = xr.open_zarr("/scratch/sadamov/pyprojects_data/aldernet/data")

my_dict = dict(
    valid_time=data.valid_time.data,
    latitude=(["y", "x"], data.latitude.data),
    longitude=(["y", "x"], data.longitude.data),
)
my_dims = {"y": data.dims.mapping["y"], "x": data.dims.mapping["x"]}

data["cos_dayofyear"] = (
    xr.DataArray(
        data=np.cos(2 * np.pi * data["valid_time.dayofyear"] / 365.25, dtype="float32")
    )
    .expand_dims(my_dims, axis=(1, 2))
    .assign_coords(coords=my_dict)
)

data["sin_dayofyear"] = (
    xr.DataArray(
        data=np.sin(2 * np.pi * data["valid_time.dayofyear"] / 365.25, dtype="float32")
    )
    .expand_dims(my_dims, axis=(1, 2))
    .assign_coords(coords=my_dict)
)

data["cos_hourofday"] = (
    xr.DataArray(
        data=np.cos(2 * np.pi * data["valid_time.hour"] / 24.0, dtype="float32")
    )
    .expand_dims(my_dims, axis=(1, 2))
    .assign_coords(coords=my_dict)
)

data["sin_hourofday"] = (
    xr.DataArray(
        data=np.sin(2 * np.pi * data["valid_time.hour"] / 24.0, dtype="float32")
    )
    .expand_dims(my_dims, axis=(1, 2))
    .assign_coords(coords=my_dict)
)

data["HSURF"] = (
    xr.open_dataarray(
        "/users/sadamov/PyProjects/aldernet/data/c1effsurf000_000",
        engine="cfgrib",
        encode_cf=("time", "geography", "vertical"),
        backend_kwargs={"filter_by_keys": {"shortName": "HSURF"}},
    )
    .drop_vars("valid_time")
    .expand_dims({"valid_time": data.dims.mapping["valid_time"]})
    .assign_coords(coords=my_dict)
    .astype("float32")
)

new_fn = "/scratch/sadamov/pyprojects_data/aldernet/data.zarr"
for i, var in enumerate(data.data_vars):
    # clear zarr encoding
    data[var].encoding.clear()
    # hardcoded chunks as "auto" does not create equal chunks for all data variables
    if i == 0:
        data[[var]].chunk({"valid_time": 32, "y": 786, "x": 1170}).persist().to_zarr(
            new_fn, mode="w"
        )
    else:
        data[[var]].chunk({"valid_time": 32, "y": 786, "x": 1170}).persist().to_zarr(
            new_fn, mode="a"
        )
