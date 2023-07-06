import numpy as np
from random import randint
from torch import nn
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scripts.finetune_evlp import load_pretrained
from datasets.evlp_xray_outcome import OutcomeDataset
from tasks.binary_classification import BinaryClassificationTask
from tasks.multiclass_classification import MulticlassClassificationTask


def saliency_map(model_path, model_backbone, label_type, image_index):
    num_label = 3 if label_type == "multiclass" else 2

    # if not backbone_saved:
    model, _ = load_pretrained(
        model_backbone=model_backbone,
        finetune_num_labels=num_label,
        trend=True,
        pretrain_num_labels=num_label,
        pretrain_path=None,
        pretrain_was_multilabel=False,
        pretrain_on_all_public_data=False,
        finetune_all=True,
    )
    if label_type == "multiclass":
        model = MulticlassClassificationTask.load_from_checkpoint(
            model_path, model=model  # if not backbone_saved else None
        )
    else:
        model = BinaryClassificationTask.load_from_checkpoint(
            model_path, model=model  # if not backbone_saved else None
        )
    model = model.model

    dataset = OutcomeDataset(
        data_dir="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/EVLP_CXR/recipient_outcome/Double/Main/",
        split="val",
        resolution=224,
        label_type="multiclass",
        trend=True,
        return_label=False,
    )
    image_to_use = dataset[image_index]
    rgb_img_1hr, rgb_img_3hr = dataset.get_original_image(image_index)
    rgb_img_1hr, rgb_img_3hr = (
        np.array(rgb_img_1hr).astype(np.float32) / 255,
        np.array(rgb_img_3hr).astype(np.float32) / 255,
    )

    model_single_tensor = WrapperModel(model)

    for timepoint in ["1hr", "3hr"]:
        if timepoint == "1hr":
            input_tensor = image_to_use[0]
            model_single_tensor.store_3hr(image_to_use[1].unsqueeze(0))
            rgb_img = rgb_img_1hr
        else:
            input_tensor = image_to_use[1]
            model_single_tensor.store_1hr(image_to_use[0].unsqueeze(0))
            rgb_img = rgb_img_3hr
        # source code: https://github.com/jacobgil/pytorch-grad-cam

        target_layers = [model_single_tensor.trend_model.feature_extractor.layer4[-1]]
        # Create an input tensor image for your model
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model_single_tensor, target_layers=target_layers)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        # for class_i in range(0, 3):
        targets = None  # Use ClassifierOutputTarget(class_i) if a specific class is to be mapped

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(
            input_tensor=input_tensor.unsqueeze(0),
            targets=targets,
            aug_smooth=True,
        )

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=0.7
        )
        visualization = Image.fromarray(visualization)
        # visualization = visualization.resize((270, 224))
        visualization.save(
            "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/inference/Adam_best_models/EVLP_saliency_"
            + timepoint
            + ".png"
        )


class WrapperModel(nn.Module):
    def __init__(self, trend_model):
        super(WrapperModel, self).__init__()
        self.trend_model = trend_model

        self.tensor1hr = None
        self.tensor3hr = None

    def store_1hr(self, image_to_use):
        self.tensor1hr = image_to_use
        self.tensor3hr = None

    def store_3hr(self, image_to_use):
        self.tensor3hr = image_to_use
        self.tensor1hr = None

    def forward(self, x):
        assert (self.tensor1hr is None and self.tensor3hr is not None) or (
            self.tensor1hr is not None and self.tensor3hr is None
        )

        if self.tensor1hr is not None:
            x_tuple = (self.tensor1hr, x)
        else:  # 1hr is None
            x_tuple = (x, self.tensor3hr)

        return self.trend_model(x_tuple)


# image_index = randint(0, 129)
saliency_map(
    model_path="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/saved_models/finetune_evlp/recipient_outcome/updatedcsv_Adam/trend_resnet50_CADLab_recipient/lightning_logs/version_8808590/checkpoints/epoch=33-step=1122.ckpt",
    model_backbone="resnet50",
    label_type="multiclass",
    image_index=7,
)
# saliency_map(
#     model_path="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/saved_models/finetune_evlp/recipient_outcome/updatedcsv_Adam/trend_resnet50_CADLab_recipient/lightning_logs/version_8808590/checkpoints/epoch=33-step=1122.ckpt",
#     model_backbone="resnet50",
#     label_type="multiclass",
#     get_1hr=False,
#     image_index=image_index,
# )
