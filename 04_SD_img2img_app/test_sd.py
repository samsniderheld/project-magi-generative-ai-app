"""A file the holds all the test functions for
the txt2img app."""
import pytest
import PIL

from model_interface import get_stable_diffusion_img2img_pipeline, change_sampler, schedulers

@pytest.fixture
def pipeline():
    """loads the txt2img pipeline
    as yield variable for pytest"""
    pipe = get_stable_diffusion_img2img_pipeline()
    yield pipe

def test_img2img(pipeline):
    """a simple test to load
    the diffusion model and generate
    an image"""
    prompt = "a puppy"
    negative_prompt = "a cat"
    input_image = PIL.Image.new(mode="RGB", size=(512, 512))
    strength = .5
    sampling_steps = 50
    cfg_scale = 7.5

    image = pipeline(prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=strength,
        num_inference_steps = sampling_steps,
        guidance_scale = cfg_scale).images[0]

    assert isinstance(image,PIL.Image.Image)
    assert image.size == input_image.size

def test_schedulers(pipeline):
    """simple test to see if schedulers
    are loading correctly"""
    for name in schedulers:
        test_scheduler = schedulers[name].from_config(pipeline.scheduler.config)
        new_pipe = change_sampler(pipeline,name)
        assert new_pipe.scheduler._class_name == test_scheduler._class_name
