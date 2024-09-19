import logging
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data.config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms

logger = logging.getLogger(__name__)


def remove_model_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


class FaceDetector:
    def __init__(
        self,
        conf_threshold=0.9,
        nms_threshold=0.4,
        top_k=250,
        imgsz=640,
        backbone="resnet50",
        backbone_weights="retinaface_resnet50.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        torch.set_grad_enabled(False)
        cudnn.benchmark = True

        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        self.cfg = cfg_re50 if backbone == "resnet50" else cfg_mnet
        self.cfg["pretrain"] = (
            False  # To avoid downloading the pretrained backbone weights
        )

        self.model = RetinaFace(cfg=self.cfg, phase="test")
        self._load_weights(pretrained_path=backbone_weights)

        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def _load_weights(self, pretrained_path):
        pretrained_state_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device)
        )

        if "state_dict" in pretrained_state_dict.keys():
            pretrained_dict = remove_model_prefix(
                pretrained_state_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = remove_model_prefix(pretrained_state_dict, "module.")

        self.model.load_state_dict(pretrained_dict, strict=False)

    def _resize_image(self, img):
        img_height, img_width, _ = img.shape
        resize_factor = 1.0
        if img_height > self.imgsz or img_width > self.imgsz:
            resize_factor = self.imgsz / max(img_height, img_width)
            img = cv2.resize(
                img,
                (int(img_width * resize_factor), int(img_height * resize_factor)),
                interpolation=cv2.INTER_LINEAR,
            )

        return img, resize_factor

    def _preprocess_image(self, img):
        img, resize_factor = self._resize_image(img)
        img = np.float32(img)
        img -= np.array([104, 117, 123], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        return img, resize_factor

    def detect_face(self, img):
        img, resize_factor = self._preprocess_image(img)
        img = img.to(self.device)
        loc, conf, landms = self.model(img)

        img_height, img_width = img.shape[2], img.shape[3]
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        scale = torch.Tensor([img_width, img_height, img_width, img_height]).to(
            self.device
        )
        boxes = boxes * scale / resize_factor
        boxes = boxes.cpu().numpy()

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale = torch.Tensor(
            [
                img_width,
                img_height,
                img_width,
                img_height,
                img_width,
                img_height,
                img_width,
                img_height,
                img_width,
                img_height,
            ]
        ).to(self.device)
        landms = landms * scale / resize_factor
        landms = landms.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # filter out the outputs with low scores
        inds = np.where(scores > self.conf_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        return dets, landms
