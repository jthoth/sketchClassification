from src.loader import DatasetBuilder, join
from tensorflow.data import TFRecordDataset
from tensorflow.keras import callbacks
from src.models import MODELS
import argparse


def main():
    parser = argparse.ArgumentParser(description='Running Settings')

    parser.add_argument('--model', help='valid options Vgg, ResNet and '
                                        'Squeeze Excitation Models',
                        required=True)

    parser.add_argument('--batch', help='# of batches', type=int,
                        default=32)

    parser.add_argument('--data', help='path where the data is stored',
                        default='data')

    args = parser.parse_args()

    if MODELS.get(args.model):
        raise ValueError("Model Does not Exist")

    builder = DatasetBuilder(args.data, shape=(256, 256))
    builder()

    data_train = TFRecordDataset(join(args.data, 'train.records'))
    data_train = data_train.map(builder.decode)
    data_train = data_train.map(builder.augmentation)
    data_train = data_train.shuffle(7000)
    data_train = data_train.batch(batch_size=args.batch)

    data_test = TFRecordDataset(join(args.data, 'test.records'))
    data_test = data_test.map(builder.decode)
    data_test = data_test.batch(batch_size=args.batch)

    model = MODELS.get(args.model)()
    model.build((1, 256, 256, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    log_dir = join('logs', args.model)
    tensor_board_callback = callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint = callbacks.ModelCheckpoint('{}.h5'.format(args.model),
                                                 save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=5,
                                            min_lr=1e-6)
    early_stop = callbacks.EarlyStopping(patience=10)

    _callbacks = [model_checkpoint, reduce_lr, early_stop,
                  tensor_board_callback]

    model.fit(data_train, epochs=100, validation_data=data_test,
              callbacks=_callbacks)


if __name__ == '__main__':
    main()
