import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # Initialize an empty list to store the output images
    output_images = []

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)


        
        # Convert the visuals to a PIL image and then to a numpy array
        for label, image in visuals.items():
            
            if label == 'fake_B_rgb':  # Assuming 'fake_B_rgb' is the generated image
                if isinstance(image, np.ndarray):  # Check if the image is a numpy array
                    # Ensure the image is in HWC format and values are in the range 0-255
                    if  image.shape[0]==3: # CHW foramt 
                    	image = image.transpose(1,2,0)
                    if np.all((image >=0) & (image < 1)): # 0<rgb.value<1 
                    	image = image * 225
                    image = (image).astype(np.uint8)  # Ensure values are uint8
                    image_pil = Image.fromarray(image)
                else:
                    print(f"Unsupported image type: {type(image)}")
                    continue

                image_resized = image_pil.resize((224, 224))  # Resize image to 224x224
                image_np = np.array(image_resized, dtype=np.uint8)
                output_images.append(image_np)
                

    webpage.save()  # save the HTML

    # Ensure we have exactly 50 images
    if len(output_images) != 50:
        print(f'Error: Expected 50 images, but got {len(output_images)}')
    else:
        # Convert list to numpy array and save it
        output_array = np.stack(output_images)
        np.save("prediction.npy", output_array)
        print('Saved prediction.npy successfully')

