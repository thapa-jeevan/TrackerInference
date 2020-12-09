import tensorflow as tf

from .generator import DataGenerator


def create_batch_generator(batch_size, num_batches, phase):
    dataset = DataGenerator(phase)

    info = {
        'idx_to_name': dataset.idx_to_name,
        'name_to_idx': dataset.name_to_idx,
        'length': len(dataset),
        'image_dir': dataset.img_dir,
        # 'anno_dir': dataset.anno_dir
    }

    if phase == 'train':
        train_dataset = tf.data.Dataset.from_generator(dataset.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        train_dataset = train_dataset.shuffle(40).batch(batch_size)

        dataset = DataGenerator('valid')
        val_dataset = tf.data.Dataset.from_generator(dataset.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info

    else:
        dataset = tf.data.Dataset.from_generator(dataset.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info