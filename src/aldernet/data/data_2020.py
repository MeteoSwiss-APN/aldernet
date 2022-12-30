"""Create a Zarr Archive based on Cosmo 2020 Data."""

# pylint: disable=R0801

# Standard library
import glob

# Third-party
import cfgrib  # type: ignore

# Remove existing zarr archive
# os.system("rm -r /scratch/sadamov/pyprojects_data/aldernet/data2020")

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
    "ALB_DIF",
    "ALB_RAD",
    "CLCT",
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

path = "/store/s83/osm/KENDA-1/ANA20/det/"
files_weather = list(set(glob.glob(path + "*")) - set(glob.glob(path + "*.*")))
files_weather.sort()
files_weather = files_weather[0 : 91 * 24]

files_cory = glob.glob(
    "/store/mch/msopr/sadamov/wd/20_cory_tuning_v3/**/lfff00*0000", recursive=True
)
files_cory.sort()
files_cory = files_cory[:]

files_alnu = glob.glob(
    "/store/mch/msopr/sadamov/wd/20_alnu_tuning_v3/**/lfff00*0000", recursive=True
)
files_alnu.sort()
files_alnu = files_alnu[:]

initialize = False
if initialize:
    ds_surface = (
        cfgrib.open_dataset(
            path + "laf2020010100",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"dataType": "fc"}},
        )
        .expand_dims({"valid_time": 1})
        .merge(
            cfgrib.open_dataset(
                path + "laf2020010100",
                encode_cf=("time", "geography", "vertical"),
                backend_kwargs={"filter_by_keys": {"dataType": "an"}},
            ).expand_dims({"valid_time": 1})
        )
        .sel({"generalVerticalLayer": 80, "generalVertical": 81})[var_selection]
    )

    ds_u = (
        cfgrib.open_dataset(
            path + "laf2020010100",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "U"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_v = (
        cfgrib.open_dataset(
            path + "laf2020010100",
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "V"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_cory = (
        cfgrib.open_dataset(files_cory[0], encode_cf=("time", "geography", "vertical"))
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_alnu = (
        cfgrib.open_dataset(files_alnu[0], encode_cf=("time", "geography", "vertical"))
        .sel({"generalVerticalLayer": 80})[["ALNU", "ALNUfr"]]
        .expand_dims({"valid_time": 1})
    )

    ds_surface = ds_surface.merge(ds_u, compat="override")
    ds_surface = ds_surface.merge(ds_v, compat="override")
    ds_surface = ds_surface.merge(ds_cory, compat="override")
    ds_surface = ds_surface.merge(ds_alnu, compat="override")

    keys = list(ds_surface.keys())
    keys.sort()
    ds_surface = ds_surface[keys]

    ds_surface.to_zarr("/scratch/sadamov/pyprojects_data/aldernet/data2020")

for file_weather, file_cory, file_alnu in zip(
    files_weather[1:], files_cory[1:], files_alnu[1:]
):
    print("CURRENT Weather FILE:", file_weather, flush=True)
    print("CURRENT Hazel FILE:", file_cory, flush=True)
    print("CURRENT Alder FILE:", file_alnu, flush=True)

    ds_surface = (
        cfgrib.open_dataset(
            file_weather,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"dataType": "fc"}},
        )
        .expand_dims({"valid_time": 1})
        .merge(
            cfgrib.open_dataset(
                file_weather,
                encode_cf=("time", "geography", "vertical"),
                backend_kwargs={"filter_by_keys": {"dataType": "an"}},
            ).expand_dims({"valid_time": 1})
        )
        .sel({"generalVerticalLayer": 80, "generalVertical": 81})[var_selection]
    )

    ds_u = (
        cfgrib.open_dataset(
            file_weather,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "U"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_v = (
        cfgrib.open_dataset(
            file_weather,
            encode_cf=("time", "geography", "vertical"),
            backend_kwargs={"filter_by_keys": {"shortName": "V"}},
        )
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_cory = (
        cfgrib.open_dataset(file_cory, encode_cf=("time", "geography", "vertical"))
        .sel({"generalVerticalLayer": 80})
        .expand_dims({"valid_time": 1})
    )

    ds_alnu = (
        cfgrib.open_dataset(file_alnu, encode_cf=("time", "geography", "vertical"))
        .sel({"generalVerticalLayer": 80})[["ALNU", "ALNUfr"]]
        .expand_dims({"valid_time": 1})
    )

    ds_surface = ds_surface.merge(ds_u, compat="override")
    ds_surface = ds_surface.merge(ds_v, compat="override")
    ds_surface = ds_surface.merge(ds_cory, compat="override")
    ds_surface = ds_surface.merge(ds_alnu, compat="override")

    keys = list(ds_surface.keys())
    keys.sort()
    ds_surface = ds_surface[keys]

    ds_surface.to_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/data2020",
        mode="a",
        append_dim="valid_time",
    )
