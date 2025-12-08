import sys
import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

# import inference code
# Get the path to the sam-3d-objects root directory (parent of sam3d_objects)
_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_package_root, "notebook"))
from inference import load_image, load_mask, check_hydra_safety, WHITELIST_FILTERS, BLACKLIST_FILTERS
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
from typing import Union, Optional
from PIL import Image

from sam3d_objects.pipeline.inference_pipeline_pointmap_sequential import InferencePipelinePointMapSequential


class InferenceSequential:
    """Sequential inference class that loads models one at a time to reduce memory usage."""

    def __init__(self, config_file: str, compile: bool = False, device: str = "cuda"):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.device = device  # set device for the pipeline
        config.workspace_dir = os.path.dirname(config_file)

        # Set device for depth model if it exists
        if "depth_model" in config and config.depth_model is not None:
            config.depth_model.device = device

        # Override _target_ to use sequential pipeline
        config._target_ = "sam3d_objects.pipeline.inference_pipeline_pointmap_sequential.InferencePipelinePointMapSequential"

        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

        # Use the sequential pipeline
        self._pipeline: InferencePipelinePointMapSequential = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=True,  # Enable mesh simplification
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def mesh_from_image_mask(image_path: str, mask_path: str, model: Optional[InferenceSequential] = None, output_path: Optional[str] = None, seed: int = 42, device: str = "cuda") -> None:
    if output_path is None:
        # Generate output path from image path: "/path/to/foo.png" -> "/path/to/completed_foo.ply"
        dir_name = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(dir_name, f"completed_{base_name}.ply")

    if model is None:
        tag = "hf"
        # Use absolute path to checkpoints directory in sam-3d-objects root
        _package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(_package_root, f"checkpoints/{tag}/pipeline.yaml")
        print("Loading model with sequential pipeline (memory-efficient mode)...")
        model = InferenceSequential(config_path, compile=False, device=device)

    image = load_image(image_path)
    mask = load_mask(mask_path)
    output = model(image, mask, seed=seed)
    output["gs"].save_ply(output_path)
    print(f"Reconstruction saved at {output_path}")

if __name__ == "__main__":
    """
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    print("Loading model with sequential pipeline (memory-efficient mode)...")
    inference = InferenceSequential(config_path, compile=False)

    # load image (RGBA only, mask is embedded in the alpha channel)
    print("Loading image and mask...")
    image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
    mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

    # run model
    print("Running inference (models will be loaded/unloaded sequentially)...")
    output = inference(image, mask, seed=42)

    # export gaussian splat
    print("Saving output...")
    output["gs"].save_ply(f"splat_sequential.ply")
    print("Your reconstruction has been saved to splat_sequential.ply")
    """
    mesh_from_image_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png", "notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png", output_path="splat_sequential.ply")
