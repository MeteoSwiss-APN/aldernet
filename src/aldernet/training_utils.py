"""Train a cGAN with UNET Architecture.

To create realistic Images of Pollen Surface Concentration Maps.
"""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Standard library
import math
import time
from pathlib import Path

# Third-party
import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from keras import layers
from keras.constraints import Constraint
from pyprojroot import here
from ray import tune
from ray.air import session

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
        W_ = tf.reshape(w, [-1, output_neurons])
        if self.u is None:
            self.u = tf.Variable(
                initial_value=tf.random_normal_initializer()(
                    shape=(output_neurons,), dtype="float32"
                ),
                trainable=False,
            )

        u_ = self.u
        v_ = None
        for _ in range(self.iterations):
            v_ = tf.matvec(W_, u_)
            v_ = tf.l2_normalize(v_)

            u_ = tf.matvec(W_, v_, transpose_a=True)
            u_ = tf.l2_normalize(u_)

        sigma = tf.tensordot(u_, tf.matvec(W_, v_, transpose_a=True), axes=1)
        self.u.assign(u_)  # '=' produces an error in graph mode
        return w / sigma


##########################


def cbr(filters, name=None):

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


def down(filters, name=None):

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

    return block


def up(filters, name=None):

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


filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]


def compile_generator(height, width, weather_features, noise_dim):
    image_input = keras.Input(shape=[height, width, 1], name="image_input")
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

        block = down(filters[ll], "down_%s-down" % ll)(block)

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

        block = up(filters[ll], "up_%s-up" % (len(filters) - ll - 1))(block)

        # Connect U-Net skip
        block = layers.Concatenate(name="up_%s-concatenate" % (len(filters) - ll - 1))(
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
    elif weather_features > 0 and noise_dim <= 0:
        return tf.keras.Model(inputs=[image_input, weather_input], outputs=pollen)
    elif weather_features <= 0 and noise_dim > 0:
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
        image = tf.concat(image, axis=1) * 10
        image = tf.cast(image, tf.uint8)
        png = tf.image.encode_png(image)
        tf.io.write_file(path, png)


##########################

bxe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]

# * https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# * Autodifferentiation vs. calculate training steps yourself
# * L1-Loss (Median) absolute value difference statt RMSE (Mean) als Loss - pixelwise
# * Calculate difference between image_b and generated ones for regression
# * Reduce output channels from 3 RGB to one pollen (+ height and width)
# * Weather input was simply zero mean and unit variance


def gan_step(
    generator,
    optimizer_gen,
    input_train,
    target_train,
    weather_train,
    noise_dim,
    weather,
):

    if noise_dim > 0:
        noise = tf.random.normal([input_train.shape[0], noise_dim])
    with tf.GradientTape() as tape_gen:
        if weather and noise_dim > 0:
            generated = generator([noise, input_train, weather_train])
        elif weather and noise_dim <= 0:
            generated = generator([input_train, weather_train])
        elif not weather and noise_dim > 0:
            generated = generator([noise, input_train])
        else:
            generated = generator([input_train])
        loss = tf.math.reduce_mean(tf.math.abs(generated - target_train))
        # loss = tf.math.reduce_mean(tf.math.squared_difference(generated, alder))
        gradients_gen = tape_gen.gradient(loss, generator.trainable_variables)

    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return loss


def train_model(
    config, generator, data_train, data_valid, run_path, noise_dim, weather
):
    mlflow.set_tracking_uri(run_path + "/mlruns")
    mlflow.set_experiment("Aldernet")
    tune_trial = tune.get_trial_name() + "/"
    Path(run_path + "/viz/" + tune_trial).mkdir(parents=True, exist_ok=True)
    Path(run_path + "/viz/valid/" + tune_trial).mkdir(parents=True, exist_ok=True)

    epoch = tf.Variable(1, dtype="int64")
    step = tf.Variable(1, dtype="int64")
    step_valid = 1

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
        if not weather:
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
                        weather,
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
                "Time taken for epoch {} is {} sec\n".format(
                    epoch.numpy(), time.time() - start
                ),
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
            session.report(
                {
                    "iterations": step,
                    "Loss_valid": loss_valid.mean(),
                    "Loss": loss_report.mean(),
                }
            )

            epoch.assign_add(1)

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
                        weather,
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
                "Time taken for epoch {} is {} sec\n".format(
                    epoch.numpy(), time.time() - start
                ),
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
                loss_valid = np.append(
                    loss_valid,
                    tf.math.reduce_mean(
                        tf.math.abs(generated_valid - alder_valid)
                    ).numpy(),
                )
                step_valid += 1
            session.report(
                {
                    "iterations": step,
                    "Loss_valid": loss_valid.mean(),
                    "Loss": loss_report.mean(),
                }
            )
            epoch.assign_add(1)


def train_model_simple(data_training, data_valid, epochs):
    model = keras.models.Sequential()
    model.add(layers.Dense(1, input_shape=(64, 128, 1)))
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

    model.fit(data_training, epochs=epochs)
    predictions = model.predict(data_valid)
    for timestep in range(
        0, math.floor(data_valid.x.shape[0] / data_valid.batch_size), 100
    ):
        write_png(
            (
                data_valid.x[timestep].values,
                data_valid.y[timestep].values,
                predictions[timestep],
            ),
            path=str(here()) + "/output/prediction" + str(timestep) + ".png",
            pretty=True,
        )
