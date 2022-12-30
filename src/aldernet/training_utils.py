"""Train a cGAN with UNET Architecture.

To create realistic Images of Pollen Surface Concentration Maps.
"""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportOptionalMemberAccess=false,reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false

# Standard library
import math
import time
from pathlib import Path

# Third-party
import keras  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import xarray as xr
from keras import layers
from keras.constraints import Constraint  # type: ignore
from keras.engine.sequential import Sequential  # type: ignore
from pyprojroot import here  # type: ignore
from ray import tune
from ray.air import Checkpoint
from ray.air import session
from tensorflow import linalg

# First-party
from aldernet.data.data_utils import Batcher


def define_filters(zoom):
    filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]

    if zoom == "":
        filters[:] = [int(x / 16) for x in filters]

    return filters


##########################


def tf_setup():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


##########################


class SpectralNormalization(Constraint):
    def __init__(self, iterations=1):
        """Define class objects."""
        self.iterations = iterations
        self.u = None

    def __call__(self, w):
        output_neurons = w.shape[-1]
        w_ = tf.reshape(w, [-1, output_neurons])
        if self.u is None:
            self.u = tf.Variable(
                initial_value=tf.random_normal_initializer()(
                    shape=(output_neurons,), dtype=tf.dtypes.float32
                ),
                trainable=False,
            )

        u_ = self.u
        v_ = None
        for _ in range(self.iterations):
            v_ = linalg.matvec(w_, u_)
            v_ = linalg.l2_normalize(v_)

            u_ = linalg.matvec(w_, v_, transpose_a=True)
            u_ = linalg.l2_normalize(u_)

        sigma = tf.tensordot(u_, linalg.matvec(w_, v_, transpose_a=True), axes=1)
        self.u.assign(u_)  # '=' produces an error in graph mode
        return w / sigma


##########################


def cbr(filters, name=None) -> Sequential:

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            # kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def down(filters, name=None) -> Sequential:

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            # kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    # block.add(layers.MaxPooling2D((2, 2), padding="same"))
    # block.add(layers.Dropout(0.05))

    return block


def up(filters, name=None) -> Sequential:

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            # kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


##########################


def compile_generator(height, width, weather_features, noise_dim, filters):
    image_input = keras.Input(shape=[height, width, 1], name="image_input")
    noise_input = weather_input = keras.Input(shape=[])
    if weather_features > 0:
        weather_input = keras.Input(
            shape=[height, width, weather_features], name="weather_input"
        )
        inputs = layers.Concatenate(name="inputs-concat")([image_input, weather_input])
    else:
        inputs = image_input
    block = cbr(filters[0], "pre-cbr-1")(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):
        block = down(filters[ll], f"down_{ll}-down")(block)
        # Collect U-Net skip connections
        u_skip_layers.append(block)
    height = block.shape[1]
    width = block.shape[2]
    if noise_dim > 0:
        noise_channels = 128
        noise_input = keras.Input(shape=noise_dim, name="noise_input")
        noise = layers.Dense(
            height * width * noise_channels,
            # kernel_constraint=SpectralNormalization()
        )(noise_input)
        noise = layers.Reshape((height, width, -1))(noise)
        block = layers.Concatenate(name="add-noise")([block, noise])
    u_skip_layers.pop()

    for ll in range(len(filters) // 2, len(filters) - 1):
        block = up(filters[ll], f"up_{(len(filters) - ll - 1)}-up")(block)
        if block.shape[slice(1, 2)] != u_skip_layers[-1].shape[slice(1, 2)]:
            block = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(
                block
            )
        # Connect U-Net skip
        block = layers.Concatenate(name=f"up_{(len(filters) - ll - 1)}-concatenate")(
            [block, u_skip_layers.pop()]
        )

    block = cbr(filters[-1], "post-cbr-1")(block)

    pollen = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="tanh",
        # kernel_constraint=SpectralNormalization(),
        name="output",
    )(block)
    if weather_features > 0 and noise_dim > 0:
        return tf.keras.Model(
            inputs=[noise_input, image_input, weather_input], outputs=pollen
        )
    elif noise_dim <= 0 < weather_features:
        return tf.keras.Model(inputs=[image_input, weather_input], outputs=pollen)
    elif weather_features <= 0 < noise_dim:
        return tf.keras.Model(inputs=[noise_input, image_input], outputs=pollen)
    else:
        return tf.keras.Model(inputs=[image_input], outputs=pollen)


##########################


def write_png(image, path, pretty):

    if pretty:

        minmin = min(image[0].min(), image[1].min(), image[2].min())
        maxmax = max(image[0].max(), image[1].max(), image[2].max())

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.1), dpi=150)
        ax1.imshow(
            image[0][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        ax2.imshow(
            image[1][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        im3 = ax3.imshow(
            image[2][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        for ax in (ax1, ax2, ax3):
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        ax1.axes.set_title("Input")
        ax2.axes.set_title("Target")
        ax3.axes.set_title("Prediction")
        fig.subplots_adjust(right=0.85, top=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im3, cax=cbar_ax)
        plt.savefig(path)
        plt.close(fig)
    else:
        image = tf.concat(image, 1) * 10
        image = tf.cast(image, tf.uint8)
        png = tf.image.encode_png(image)
        tf.io.write_file(path, png)


##########################

bxe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# * https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# * Autodifferentiation vs. calculate training steps yourself
# * L1-Loss (Median) absolute value difference statt RMSE (Mean) also Loss - pixelwise
# * Calculate difference between image_b and generated ones for regression
# * Reduce output channels from 3 RGB to one pollen (+ height and width)
# * Weather input was simply zero mean and unit variance


def gan_step(  # pylint: disable=R0913
    generator,
    optimizer_gen,
    input_train,
    target_train,
    weather_train,
    noise_dim,
    add_weather,
):
    noise = tf.random.normal(shape=[])
    if noise_dim > 0:
        noise = tf.random.normal([input_train.shape[0], noise_dim])
    with tf.GradientTape() as tape_gen:
        if add_weather and noise_dim > 0:
            generated = generator([noise, input_train, weather_train])
        elif add_weather and noise_dim <= 0:
            generated = generator([input_train, weather_train])
        elif not add_weather and noise_dim > 0:
            generated = generator([noise, input_train])
        else:
            generated = generator([input_train])
        loss = tf.math.reduce_mean(tf.math.abs(generated - target_train))
        # loss = tf.math.reduce_mean(tf.math.squared_difference(generated, alder))
        gradients_gen = tape_gen.gradient(loss, generator.trainable_variables)

    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return loss


def train_model(  # pylint: disable=R0912,R0913,R0914,R0915
    config, generator, data_train, data_valid, run_path, noise_dim, add_weather, shuffle
):

    data_train = Batcher(
        data_train, batch_size=32, add_weather=add_weather, shuffle=shuffle
    )
    data_valid = Batcher(
        data_valid, batch_size=32, add_weather=add_weather, shuffle=shuffle
    )

    mlflow.set_tracking_uri(run_path + "/mlruns")
    mlflow.set_experiment("Aldernet")
    tune_trial = tune.get_trial_name() + "/"  # type:ignore
    Path(run_path + "/viz/" + tune_trial).mkdir(parents=True, exist_ok=True)
    Path(run_path + "/viz/valid/" + tune_trial).mkdir(parents=True, exist_ok=True)

    epoch = tf.Variable(1, dtype="int64")
    step = tf.Variable(1, dtype="int64")
    step_valid = 1
    checkpoint = Checkpoint.from_dict(dict({"dummy": 0}))

    # betas need to be floats, or checkpoint restoration fails
    optimizer_gen = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"],
        beta_1=config["beta_1"],
        beta_2=config["beta_2"],
        epsilon=1e-08,
    )

    while True:
        start = time.time()
        loss_report = np.zeros(0)
        loss_valid = np.zeros(0)
        if not add_weather:
            for i in range(math.floor(data_train.x.shape[0] / data_train.batch_size)):

                hazel_train = data_train[i][0]
                alder_train = data_train[i][1]

                print(epoch.numpy(), "-", step.numpy(), flush=True)

                loss_report = np.append(
                    loss_report,
                    gan_step(
                        generator,
                        optimizer_gen,
                        hazel_train,
                        alder_train,
                        None,
                        noise_dim,
                        add_weather,
                    ).numpy(),
                )
                if noise_dim > 0:
                    noise_train = tf.random.normal([hazel_train.shape[0], noise_dim])
                    generated_train = generator([noise_train, hazel_train])
                else:
                    generated_train = generator([hazel_train])
                index = np.random.randint(hazel_train.shape[0])

                viz = (
                    hazel_train[index],
                    alder_train[index],
                    generated_train[index].numpy(),
                )

                write_png(
                    viz,
                    run_path
                    + "/viz/"
                    + tune_trial
                    + str(epoch.numpy())
                    + "-"
                    + str(step.numpy())
                    + ".png",
                    pretty=True,
                )

                step.assign_add(1)

            print(
                f"Time taken for epoch {epoch.numpy()} is {time.time() - start} sec\n",
                flush=True,
            )

            for i in range(math.floor(data_valid.x.shape[0] / data_valid.batch_size)):

                hazel_valid = data_valid[i][0]
                alder_valid = data_valid[i][1]

                if noise_dim > 0:
                    noise_valid = tf.random.normal([hazel_valid.shape[0], noise_dim])
                    generated_valid = generator([noise_valid, hazel_valid])
                else:
                    generated_valid = generator([hazel_valid])
                index = np.random.randint(hazel_valid.shape[0])
                viz = (
                    hazel_valid[index],
                    alder_valid[index],
                    generated_valid[index].numpy(),
                )
                write_png(
                    viz,
                    run_path
                    + "/viz/valid/"
                    + tune_trial
                    + str(epoch.numpy())
                    + "-"
                    + str(step_valid)
                    + ".png",
                    pretty=True,
                )
                step_valid += 1
                loss_valid = np.append(
                    loss_valid,
                    tf.math.reduce_mean(
                        tf.math.abs(generated_valid - alder_valid)
                    ).numpy(),
                )

                checkpoint = Checkpoint.from_dict(
                    dict(
                        epoch=epoch,
                        model=generator,
                    )
                )

            session.report(
                {
                    "iterations": step,
                    "Loss_valid": loss_valid.mean(),
                    "Loss": loss_report.mean(),
                },
                checkpoint=checkpoint,
            )
            epoch.assign_add(1)
            if shuffle:
                data_train.on_epoch_end()
                data_valid.on_epoch_end()

        else:
            start = time.time()
            for i in range(math.floor(data_train.x.shape[0] / data_train.batch_size)):

                hazel_train = data_train[i][0]
                weather_train = data_train[i][1]
                alder_train = data_train[i][2]

                print(epoch.numpy(), "-", step.numpy(), flush=True)

                loss_report = np.append(
                    loss_report,
                    gan_step(
                        generator,
                        optimizer_gen,
                        hazel_train,
                        alder_train,
                        weather_train,
                        noise_dim,
                        add_weather,
                    ).numpy(),
                )
                if noise_dim > 0:
                    noise_train = tf.random.normal([hazel_train.shape[0], noise_dim])
                    generated_train = generator(
                        [noise_train, hazel_train, weather_train]
                    )
                else:
                    generated_train = generator([hazel_train, weather_train])
                index = np.random.randint(hazel_train.shape[0])

                viz = (
                    hazel_train[index],
                    alder_train[index],
                    generated_train[index].numpy(),
                )
                write_png(
                    viz,
                    run_path
                    + "/viz/"
                    + tune_trial
                    + str(epoch.numpy())
                    + "-"
                    + str(step.numpy())
                    + ".png",
                    pretty=True,
                )

                step.assign_add(1)

            print(
                f"Time taken for epoch {epoch.numpy()} is {time.time() - start} sec\n",
                flush=True,
            )

            for i in range(math.floor(data_valid.x.shape[0] / data_valid.batch_size)):

                hazel_valid = data_valid[i][0]
                weather_valid = data_valid[i][1]
                alder_valid = data_valid[i][2]

                if noise_dim > 0:
                    noise_valid = tf.random.normal([hazel_valid.shape[0], noise_dim])
                    generated_valid = generator(
                        [noise_valid, hazel_valid, weather_valid]
                    )
                else:
                    generated_valid = generator([hazel_valid, weather_valid])
                index = np.random.randint(hazel_valid.shape[0])
                viz = (
                    hazel_valid[index],
                    alder_valid[index],
                    generated_valid[index].numpy(),
                )
                write_png(
                    viz,
                    run_path
                    + "/viz/valid/"
                    + tune_trial
                    + str(epoch.numpy())
                    + "-"
                    + str(step_valid)
                    + ".png",
                    pretty=True,
                )
                step_valid += 1
                loss_valid = np.append(
                    loss_valid,
                    tf.math.reduce_mean(
                        tf.math.abs(generated_valid - alder_valid)
                    ).numpy(),
                )

                checkpoint = Checkpoint.from_dict(dict(epoch=epoch, model=generator))

            session.report(
                {
                    "iterations": step,
                    "Loss_valid": loss_valid.mean(),
                    "Loss": loss_report.mean(),
                },
                checkpoint=checkpoint,
            )
            epoch.assign_add(1)
            if shuffle:
                data_train.on_epoch_end()
                data_valid.on_epoch_end()


def train_model_simple(  # pylint: disable=R0914,R0915
    data_train, data_valid, epochs, add_weather, conv=True
):

    if add_weather:
        data_train.x = xr.concat([data_train.x, data_train.weather], dim="var")
        data_valid.x = xr.concat([data_valid.x, data_valid.weather], dim="var")
    inputs = keras.Input(
        shape=[data_train.x.shape[1], data_train.x.shape[2], data_train.x.shape[3]]
    )
    # Build U-Net model
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    if conv:
        c1 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(c1)

        c2 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(c2)

        c3 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(c3)

        c4 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(c4)

        c5 = tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c5)

        u6 = tf.keras.layers.Conv2DTranspose(
            128, (2, 2), strides=(2, 2), padding="same"
        )(c5)
        if u6.shape != c4.shape:
            u6 = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(u6)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c6)

        u7 = tf.keras.layers.Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding="same"
        )(c6)
        if u7.shape != c3.shape:
            u7 = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(u7)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c7)

        u8 = tf.keras.layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding="same"
        )(c7)
        if u8.shape != c2.shape:
            u8 = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(u8)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c8)

        u9 = tf.keras.layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding="same"
        )(c8)
        if u9.shape != c1.shape:
            u9 = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(u9)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation=tf.keras.activations.elu,
            kernel_initializer="he_normal",
            padding="same",
        )(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    else:
        model = keras.models.Sequential()
        model.add(
            layers.Dense(
                1,
                input_shape=(
                    [
                        data_train.x.shape[1],
                        data_train.x.shape[2],
                        data_train.x.shape[3],
                    ]
                ),
            )
        )
        model.add(
            layers.Conv2D(
                filters=2,
                kernel_size=4,
                strides=1,
                padding="same",
                use_bias=False,
                # kernel_constraint=SpectralNormalization()
            )
        )
        model.add(layers.Dense(1, activation="linear"))
    model.summary()

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )

    model.fit(data_train, epochs=epochs)
    predictions = model.predict(data_valid)
    for timestep in range(0, predictions.shape[0], 100):
        write_png(
            (
                data_valid.x[timestep].values,
                data_valid.y[timestep].values,
                predictions[timestep],
            ),
            path=str(here()) + "/output/prediction" + str(timestep) + ".png",
            pretty=True,
        )
    return model
