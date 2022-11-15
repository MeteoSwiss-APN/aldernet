"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import mlflow
import xarray as xr
from keras.utils import plot_model
from pyprojroot import here
from ray import init
from ray import shutdown
from ray import tune
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from tensorflow import random

# First-party
from aldernet.data.data_utils import Batcher
from aldernet.training_utils import compile_generator
from aldernet.training_utils import tf_setup
from aldernet.training_utils import train_model
from aldernet.training_utils import train_model_simple

# ---> DEFINE SETTINGS HERE <--- #
tune_with_ray = True
noise_dim = 0
add_weather = False
# -------------------------------#
if not tune_with_ray:
    add_weather = False


tf_setup()
random.set_seed(1)

run_path = str(here()) + "/output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if tune_with_ray:
    Path(run_path + "/viz/valid").mkdir(parents=True, exist_ok=True)

data_train = xr.open_zarr("/scratch/sadamov/aldernet/data_zoom/data_train.zarr")
data_valid = xr.open_zarr("/scratch/sadamov/aldernet/data_zoom/data_valid.zarr")

batcher_train = Batcher(data_train, batch_size=32, weather=add_weather)
batcher_valid = Batcher(data_valid, batch_size=32, weather=add_weather)

if tune_with_ray:
    height = data_train.CORY.shape[1]
    width = data_train.CORY.shape[2]
    if add_weather:
        weather_features = len(data_train.drop_vars(("ALNU", "CORY")).data_vars)
    else:
        weather_features = 0
    generator = compile_generator(height, width, weather_features, noise_dim)

    with open(run_path + "/generator_summary.txt", "w") as handle:
        with redirect_stdout(handle):
            generator.summary()
    plot_model(generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96)

    # Train

    # Use hyperparameter search functionality by ray tune and log experiment
    shutdown()
    init(
        runtime_env={
            "working_dir": str(here()),
            "excludes": ["data/", ".git/", "images/"],
        }
    )

    tune.run(
        tune.with_parameters(
            train_model,
            generator=generator,
            data_train=batcher_train,
            data_valid=batcher_valid,
            run_path=run_path,
            noise_dim=noise_dim,
            weather=add_weather,
        ),
        metric="Loss",
        num_samples=1,
        resources_per_trial={"gpu": 1},  # Choose approriate Device
        stop={"training_iteration": 20},
        config={
            # define search space here
            "learning_rate": tune.choice([0.0001]),
            "beta_1": tune.choice([0.85]),
            "beta_2": tune.choice([0.97]),
            "batch_size": tune.choice([40]),
            "mlflow": {
                "experiment_name": "Aldernet",
                "tracking_uri": mlflow.get_tracking_uri(),
            },
        },
        local_dir=run_path,
        keep_checkpoints_num=1,
        checkpoint_score_attr="Loss",
        checkpoint_at_end=True,
        callbacks=[
            MLflowLoggerCallback(
                experiment_name="Aldernet",
                tracking_uri=run_path + "/mlruns",
                save_artifact=True,
            )
        ],
    )
    # rsync commands to merge the mlruns directories
    rsync_cmd = "rsync" + " -avzh " + run_path + "/mlruns" + " " + str(here())
    subprocess.run(rsync_cmd, shell=True)
else:
    train_model_simple(batcher_train, batcher_valid, epochs=10)
