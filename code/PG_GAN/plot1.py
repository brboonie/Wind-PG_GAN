import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors, colorbar
from matplotlib import gridspec, pyplot as plt
import numpy as np
import data_utils
import train

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_data_and_models(scenes_fn, model_name="cs_modis_cgan-release",
    epoch=90, scene_size=64, modis_var_dim=2, noise_dim=64,
    lr_disc=0.0001, lr_gan=0.0002):


    scenes = data_utils.load_cloudsat_scenes(scenes_fn, shuffle_seed=214101)

    (gen, disc, gan, opt_disc, opt_gan) = train.create_models(scene_size, modis_var_dim, lr_disc, lr_gan)

    train.load_model_state(gen, disc, gan, opt_disc, opt_gan, model_name, epoch)

    return (scenes, gen, disc, gan)

def generate_scenes(gen, modis_vars, modis_mask):

    scene_gen = gen.predict([modis_vars])
    recon_image = (1 - modis_mask) * scene_gen + modis_mask * modis_vars

    return recon_image


def plot_samples_cmp(gen,  modis_vars, modis_mask):

    scene_gen = [None]*1
    scene_gen = generate_scenes(gen, modis_vars, modis_mask)
    scene_gen = data_utils.rescale_scene(scene_gen)
    # scene_real = data_utils.rescale_scene(scene_real)
    scene_initial = data_utils.rescale_scene(modis_vars)

    # scene_dispre = scene_real - scene_gen[-1]
    print(scene_gen.shape)

    # 保存scene_gen和scene_real为npz文件
    np.savez(dir_path+"/../data/scenes_output.npz", scene_initial=scene_initial, scene_gen=scene_gen)




def plot_all(scenes_fn, model_name="cs_modis_cgan-release"):
    (scenes, gen, disc, gan) = load_data_and_models(scenes_fn,
        model_name=model_name)


    # (scene_real, modis_vars, modis_mask) = scenes["train"]

    (modis_vars, modis_mask) = scenes["train"]

    # plot_samples_cmp(gen, scene_real, modis_vars, modis_mask)
    plot_samples_cmp(gen, modis_vars, modis_mask)


    # plot_gen_vary(gen, modis_vars, modis_mask)
    # plt.savefig("../figures/gen_vary.pdf", bbox_inches='tight')
    # plt.close()