from tensorflow.data import TFRecordDataset
from unittest import TestCase
from src import models, loader


class TestModelStructure(TestCase):

    def test_vgg_model_structure_should_be_ok(self):
        expected_layers = 27
        model = models.VggModel()
        model.build((1, 256, 256, 3))
        total = len(model.information_flow)
        self.assertEqual(total, expected_layers)

    def test_dataset_train_test_builder_should_be_ok(self):
        builder = loader.DatasetBuilder('data')
        # builder()

    def test_loader_tf_data_set_should_be_ok(self):
        builder = loader.DatasetBuilder('data', shape=(256, 256))
        dataset = TFRecordDataset('data/test.records')
        dataset = dataset.map(builder.decoder)
        dataset = dataset.map(builder.data_augmentation)
        dataset = dataset.shuffle(4000)
        dataset = dataset.batch(batch_size=60)