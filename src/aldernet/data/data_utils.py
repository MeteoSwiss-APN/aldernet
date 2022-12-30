"""Helper functions to import and pre-process zarr archives."""

# Standard library
import math

# Third-party
import numpy as np
import tensorflow as tf  # type: ignore

select_params = [
    "ALNU",
    "CORY",
    "CORYctsum",
    "CORYfe",
    "CORYfr",
    "CORYrprec",
    "CORYsaisn",
    "CORYsdes",
    "cos_dayofyear",
    "cos_hourofday",
    "FIS",
    "HPBL",
    "HSURF",
    "QR",
    "P",
    "sin_dayofyear",
    "sin_hourofday",
    "TQC",
    "U",
    "V",
]


class Batcher(tf.keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, data, batch_size, add_weather, shuffle=True):
        """Initialize."""
        self.x = data[["CORY"]].to_array("var").transpose("valid_time", ..., "var")
        if add_weather:
            self.weather = (
                data[select_params]
                .drop_vars(("ALNU", "CORY"))
                .to_array("var")
                .transpose("valid_time", ..., "var")
            )
        self.y = data[["ALNU"]].to_array("var").transpose("valid_time", ..., "var")
        self.batch_size = batch_size
        self.add_weather = add_weather
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.add_weather:
            batch_weather = self.weather[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            return batch_x.values, batch_weather.values, batch_y.values
        else:
            return batch_x.values, batch_y.values

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        if self.shuffle is True:
            idx = np.random.permutation(len(self.x.values))
            self.x.values = self.x.values[idx]
            self.y.values = self.y.values[idx]
            if self.add_weather:
                self.weather.values = self.weather.values[idx]
            print("Data Reshuffled!", flush=True)
