import types
import torch
import cv2
import os
import numpy as np
import collections

from random import shuffle
from pathlib import Path
import albumentations as A

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torch.utils.dlpack as torch_dlpack

class ExternalInputIterator(object):
    def __init__(self, batch_size, data_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.files = [ f.path for f in os.scandir(self.data_dir) if f.is_dir() ]
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []

        for _ in range(self.batch_size):
            batch.append(self.files[self.i])
            self.i = (self.i + 1) % self.n

        return batch

    next = __next__

class SequenceSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id, size):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.data_iterator = data_iterator

        self.source = ops.SequenceReader(device="cpu", file_root)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id, size, eval_enabled=False):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.data_iterator = data_iterator
        self.eval_enabled = eval_enabled

        self.ops.SequenceReader(device="cpu", file_root)

        
        self.iim1 = ops.ExternalSource()
        self.iim2 = ops.ExternalSource()
        self.iim3 = ops.ExternalSource()
        
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.int = ops.Resize(device="gpu", resize_x=size[0]*2, resize_y=size[1]*2, image_type=types.RGB, interp_type=types.INTERP_CUBIC)

        self.res = ops.Resize(device="gpu", resize_x=size[0], resize_y=size[1], image_type=types.RGB, interp_type=types.INTERP_LINEAR)
        self.down = ops.Resize(device="gpu", resize_x=size[0]//2, resize_y=size[1]//2, image_type=types.RGB, interp_type=types.INTERP_LANCZOS3)
        self.up = ops.Resize(device="gpu", resize_x=size[0], resize_y=size[1], image_type=types.RGB, interp_type=types.INTERP_CUBIC)

        self.rotate = ops.Rotate(device='gpu', interp_type=types.INTERP_LINEAR, keep_size=True)
        self.uniform = ops.Uniform(range = (-50., 50.))

        self.finish = ops.CropMirrorNormalize(device="gpu", std=255., mean=0., output_dtype=types.FLOAT, image_type=types.RGB)

    def augmentation(self, dlpacks):
        tensors = torch.stack([torch_dlpack.from_dlpack(dlpack) for dlpack in dlpacks]).to('cpu')

        output = [self.seq(image=np.array(tensors[i]))['image'] for i in range(len(dlpacks))]
        output = torch.from_numpy(np.stack(output)).to('cuda')
    
        return [torch_dlpack.to_dlpack(tensor) for tensor in output]

    def define_graph(self):
        self.im1 = self.iim1()
        self.im2 = self.iim2()
        self.im3 = self.iim3()

        im1 = self.decode(self.im1)
        im2 = self.decode(self.im2)
        im3 = self.decode(self.im3)

        if not self.eval_enabled:
            im1 = self.res(im1)
            im2 = self.res(im2)
            im3 = self.res(im3)

            angle = self.uniform()
            im1 = self.rotate(im1, angle=angle)
            im2 = self.rotate(im2, angle=angle)
            im3 = self.rotate(im3, angle=angle)

            t = im2
            
            im1 = self.down(im1)
            im2 = self.down(im2)
            im3 = self.down(im3)
        else:
            t = self.int(im2)

            im1 = self.res(im1)
            im2 = self.res(im2)
            im3 = self.res(im3)
        
        im1 = self.finish(im1)
        im2 = self.finish(im2)
        im3 = self.finish(im3)
        targets = self.finish(t)

        return (im1, im2, im3, targets)

    def iter_setup(self):
        (im1, im2, im3) = self.data_iterator.next()
        self.feed_input(self.im1, im1)
        self.feed_input(self.im2, im2)
        self.feed_input(self.im3, im3)
