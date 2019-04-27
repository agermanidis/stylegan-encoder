import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import dnnlib
import runway
from runway.data_types import number, vector, image
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from tqdm import tqdm

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

@runway.setup
def setup():
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, 1, randomize_noise=False)
    perceptual_model = PerceptualModel(256, layer=9, batch_size=1)
    perceptual_model.build_perceptual_model(generator.generated_image)
    return perceptual_model, generator


INPUTS = {
    'reference': image,
    'noise': vector(512),
    'iterations': number(default=1000, min=1, max=10000),
    'learning_rate': number(default=1, step=0.01, min=0, max=3)
}


@runway.command('autoencode', inputs=INPUTS, outputs={'output': image})
def autoencode(model, inputs):
    perceptual_model, generator = model
    perceptual_model.set_reference_images([inputs['reference']])
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs['iterations'], learning_rate=inputs['learning_rate'])
    pbar = tqdm(op, leave=False, total=inputs['iterations'])
    for loss in pbar:
        pbar.set_description('Loss: %.2f' % loss)
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    generator.reset_dlatents()
    return generated_images[0]


if __name__ == '__main__':
    runway.run()
