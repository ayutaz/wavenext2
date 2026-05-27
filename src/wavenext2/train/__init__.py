"""wavenext2.train public API."""

from wavenext2.train.train_gan import TrainState, main, train_gan_step

__all__ = ["TrainState", "main", "train_gan_step"]
