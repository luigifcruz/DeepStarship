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
        #self.files = list(Path(self.data_dir).rglob("*.[jJpP][pPnN][gG]"))
        self.files = []
        self.dirs = [ f.path for f in os.scandir(self.data_dir) if f.is_dir() ]
        for d in self.dirs:
            self.files.extend([ f.path for f in os.scandir(d) if f.is_dir() ])
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        im1_batch = []
        im2_batch = []
        im3_batch = []

        for _ in range(self.batch_size):
            jpeg_path = self.files[self.i]

            im1 = os.path.join(jpeg_path, "im1.png")
            im2 = os.path.join(jpeg_path, "im2.png")
            im3 = os.path.join(jpeg_path, "im3.png")

            im1_batch.append(np.frombuffer(open(im1, 'rb').read(), dtype=np.uint8))
            im2_batch.append(np.frombuffer(open(im2, 'rb').read(), dtype=np.uint8))
            im3_batch.append(np.frombuffer(open(im3, 'rb').read(), dtype=np.uint8))            

            self.i = (self.i + 1) % self.n

        return (im1_batch, im2_batch, im3_batch)

    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id, size, eval_enabled=False):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False)
        self.data_iterator = data_iterator
        self.eval_enabled = eval_enabled
        
        self.iim1 = ops.ExternalSource()
        self.iim2 = ops.ExternalSource()
        self.iim3 = ops.ExternalSource()
        
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.rint = ops.Reshape(device="gpu", rel_shape=[1, 1, -1])

        self.int = ops.Resize(device="gpu", resize_x=size[0]*2, resize_y=size[1]*2, image_type=types.RGB, interp_type=types.INTERP_CUBIC)

        self.res = ops.Resize(device="gpu", resize_x=size[0], resize_y=size[1], image_type=types.RGB, interp_type=types.INTERP_LINEAR)
        self.down = ops.Resize(device="gpu", resize_x=size[0]//2, resize_y=size[1]//2, image_type=types.RGB, interp_type=types.INTERP_GAUSSIAN)
        self.up = ops.Resize(device="gpu", resize_x=size[0], resize_y=size[1], image_type=types.RGB, interp_type=types.INTERP_CUBIC)
        
        self.seq = A.Compose({
            A.ElasticTransform(alpha=25, sigma=500, alpha_affine=1, approximate=True, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1),
            A.GaussNoise(p=1),
        })
        self.aug = ops.DLTensorPythonFunction(device="gpu", function=self.augmentation)

        self.cmn = ops.CropMirrorNormalize(device="gpu", std=255., mean=0., output_dtype=types.FLOAT, image_type=types.RGB)

        self.rotate = ops.Rotate(device='gpu', interp_type=types.INTERP_LINEAR, keep_size=True)
        self.uniform = ops.Uniform(range = (-50., 50.))

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
            
            im1 = self.up(self.down(self.aug(im1)))
            im2 = self.up(self.down(self.aug(im2)))
            im3 = self.up(self.down(self.aug(im3)))
        else:
            im1 = self.int(im1)
            im2 = self.int(im2)
            im3 = self.int(im3)

            t = self.int(im2)
        
        im1 = self.cmn(im1)
        im2 = self.cmn(im2)
        im3 = self.cmn(im3)
        targets = self.cmn(t)

        return (im1, im2, im3, targets)

    def iter_setup(self):
        (im1, im2, im3) = self.data_iterator.next()
        self.feed_input(self.im1, im1)
        self.feed_input(self.im2, im2)
        self.feed_input(self.im3, im3)
