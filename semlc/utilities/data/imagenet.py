import glob
import os
from typing import List

import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.types as types
from matplotlib import pyplot as plt
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class DALITorchLoader(DALIClassificationIterator):

    def __init__(self, pipelines, reader_name, auto_reset, last_batch_policy, n_classes, dataset_name):
        super().__init__(pipelines, reader_name=reader_name, auto_reset=auto_reset, last_batch_policy=last_batch_policy)
        self.n_classes = n_classes
        self.dataset_name = dataset_name

    def __next__(self):
        data = super().__next__()[0]
        return data["data"], data["label"]


def imagenet_dali_dataloader(
        tfrecord_filenames: List[str],
        tfrecord_idx_filenames: List[str],
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 128,
        num_threads: int = os.cpu_count(),
        image_size: int = 224,
        prefetch: int = 1,
        training: bool = True,
        gpu: bool = True) -> DALITorchLoader:
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=0)

    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrecord_filenames,
            index_path=tfrecord_idx_filenames,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=10000,
            read_ahead=True,
            prefetch_queue_depth=prefetch,
            name='Reader',
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
            })

        jpegs = inputs["image/encoded"]
        if training:
            images = fn.decoders.image_random_crop(
                jpegs,
                device='mixed' if gpu else 'cpu',
                output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
            images = fn.resize(images,
                               device='gpu' if gpu else 'cpu',
                               resize_x=image_size,
                               resize_y=image_size,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs,
                                       device='mixed' if gpu else 'cpu',
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device='gpu' if gpu else 'cpu',
                               size=int(image_size / 0.875),
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            crop=(image_size, image_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror)
        label = inputs["image/class/label"] - 1  # 0-999
        label = fn.element_extract(label, element_map=0)  # Flatten
        label = label.gpu()
        pipe.set_outputs(images, label)

    pipe.build()

    # build the dataloader
    loader = DALITorchLoader(
        [pipe], reader_name="Reader",
        auto_reset=True,
        last_batch_policy='DROP' if training else 'PARTIAL',
        n_classes=1000,
        dataset_name="ImageNet")

    return loader


if __name__ == "__main__":
    train_recs = '../../../data/imagenet/train/*'
    val_recs = '../../../data/imagenet/validation/*'
    train_idx = '../../../data/imagenet/idx_files/train/*'
    val_idx = '../../../data/imagenet/idx_files/validation/*'

    train_loader = imagenet_dali_dataloader(
        sorted(glob.glob(train_recs)),
        sorted(glob.glob(train_idx)),
        batch_size=128,
        training=True)

    for (data,) in train_loader:
        data, target = data['data'], data['label']
        print(data.shape)
        plt.imshow(data[0].cpu())
        plt.suptitle(target[0].cpu())
        plt.show()
