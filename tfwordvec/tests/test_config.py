import tensorflow as tf
from ..config import build_config


class TestBuildConfig(tf.test.TestCase):
    def test_empty_overrides(self):
        c = build_config({})

    def test_addons_optimizer(self):
        build_config({'train_optim': 'Addons>RectifiedAdam'})


if __name__ == "__main__":
    tf.test.main()
