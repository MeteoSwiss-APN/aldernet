"""Create a Zarr Archive based on Cosmo 2022 Data."""

# pylint: disable=R0801

# Standard library
import glob

# Third-party
import cfgrib  # type: ignore

# Remove existing zarr archive
# os.system("rm -r /scratch/sadamov/pyprojects_data/aldernet/data2022")

# Variables to be extracted from surface-level GRIB file
# Unfortunately, all variables have to be read first (metadata only - lazily),
# as filtered import only supports one shortName at a time
var_selection = [
    "T",
    "P",
    "QV",
    "QC",
    "QI",
    "QR",
    "QS",
    "QG",
    "PP",
    "CLC",
    "CORY",
    "ALNU",
    "ALB_DIF",
    "ALB_RAD",
    "ALNUfr",
    "CLCT",
    "CORYctsum",
    "CORYfe",
    "CORYfr",
    "CORYhcem",
    "CORYreso",
    "CORYress",
    "CORYrprec",
    "CORYsaisa",
    "CORYsaisn",
    "CORYsdes",
    "CORYtthre",
    "CORYtthrs",
    "CORYtune",
    "DPSDT",
    "DURSUN",
    "FIS",
    "FOR_D",
    "FOR_E",
    "FR_LAND",
    "HPBL",
    "HSURF",
    "LAI",
    "PLCOV",
    "PS",
    "ROOTDP",
    "SKYVIEW",
    "SLO_ANG",
    "SLO_ASP",
    "SOILTYP",
    "TCH",
    "TCM",
    "TQC",
    "TQV",
    "TWATER",
    "T_G",
]
var_selection.sort()

path = "/store/s83/osm/KENDA-1/ANA22/det/"
files = list(set(glob.glob(path + "*")) - set(glob.glob(path + "*.*")))
files.sort()
files_red = files[39 * 24 + 1 : 90 * 24]
# files_red = files[39 * 24 + 1 : 39 * 24 + 4]

initialize = False
if initialize:
    ds_surface = (
        cfgrib.open_dataset(
            path + "laf2022020900",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"dataType": "fc"}},
        )
        .expand_dims({"valid_time": 1})
        .merge(
            cfgrib.open_dataset(
                path + "laf2022020900",
                encode_cf=("time", "geography", "vertical"),
                backend_kwargs={"filter_by_keys": {"dataType": "an"}},
            ).expand_dims({"valid_time": 1})
        )
        .sel({"generalVerticalLayer": 80, "generalVertical": 81})[var_selection]
    )

    ds_u = (
        cfgrib.open_dataset(
            path + "laf2022020900",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "U"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_v = (
        cfgrib.open_dataset(
            path + "laf2022020900",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "V"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_surface = ds_surface.merge(ds_u, compat="override")
    ds_surface = ds_surface.merge(ds_v, compat="override")

    keys = list(ds_surface.keys())
    keys.sort()
    ds_surface = ds_surface[keys]

    ds_surface.to_zarr("/scratch/sadamov/pyprojects_data/aldernet/data2022")

for file in files_red:
    print("CURRENT FILE:", file, flush=True)

    ds_surface = (
        cfgrib.open_dataset(
            file,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"dataType": "fc"}},
        )
        .expand_dims({"valid_time": 1})
        .merge(
            cfgrib.open_dataset(
                file,
                encode_cf=("time", "geography", "vertical"),
                backend_kwargs={"filter_by_keys": {"dataType": "an"}},
            ).expand_dims({"valid_time": 1})
        )
        .sel({"generalVerticalLayer": 80, "generalVertical": 81})[var_selection]
    )

    ds_u = (
        cfgrib.open_dataset(
            file,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "U"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_v = (
        cfgrib.open_dataset(
            file,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "V"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_surface = ds_surface.merge(ds_u, compat="override")
    ds_surface = ds_surface.merge(ds_v, compat="override")

    keys = list(ds_surface.keys())
    keys.sort()
    ds_surface = ds_surface[keys]

    ds_surface.to_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/data2022",
        mode="a",
        append_dim="valid_time",
    )
