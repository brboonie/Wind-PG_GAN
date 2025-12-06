import gc
import numpy as np
import h5py
import keras.backend as K
import netCDF4
from tensorflow.keras.optimizers import Adam

def load_cloudsat_scenes(fn, n=None, right_handed=False, frac_validate=0.1, shuffle=True, shuffle_seed=None):

    with netCDF4.Dataset(fn, 'r') as ds:
        if n is None:
            n = ds["u10_normalized_wunan"].shape[0]
        #     obs删
        cs_scenes_u = np.array(ds["u10_normalized_com"][:n,:,:])
        cs_scenes_u = cs_scenes_u.reshape(cs_scenes_u.shape+(1,))
        cs_scenes_v = np.array(ds["v10_normalized_com"][:n,:,:])
        cs_scenes_v = cs_scenes_v.reshape(cs_scenes_v.shape+(1,))
        wind_scenes = np.concatenate((cs_scenes_u, cs_scenes_v), axis=-1)
        print(wind_scenes.shape)


        modis_vars = np.zeros((n,)+ds["u10_normalized_wunan"].shape[1:]+(2,), dtype=np.float32)
        modis_vars[:,:,:,0] = ds["u10_normalized_wunan"][:n,:,:]
        modis_vars[:,:,:,1] = ds["v10_normalized_wunan"][:n,:,:]
        print(modis_vars.shape)

        modis_mask = np.zeros((n,)+ds["v10_normalized_wunan"].shape[1:]+(1,), dtype=np.float32)
        modis_mask[:,:,:,0] = ds["era_mask_1"][:n,:,:]
        print(modis_mask.shape)

    num_scenes = modis_mask.shape[0]

    # 清理内存
    gc.collect()

    num_train = int(round(num_scenes*(1.0-frac_validate)))

    scenes = {
        "train": (
            wind_scenes[:num_train,...],
            modis_vars[:num_train,...],
            modis_mask[:num_train,...]
        ),
        "validate": (
            wind_scenes[num_train:,...],
            modis_vars[num_train:,...], 
            modis_mask[num_train:,...]
        )
    }
    # (8760,64,64,1)
    return scenes


def load_physical_data(pn,n=None, right_handed=False, frac_validate=0.1, shuffle=True, shuffle_seed=None ):
    with netCDF4.Dataset(pn, 'r') as ds:
        if n is None:
            n = ds["v10_normalized_com"].shape[0]
        u10 = np.array(ds["u10_normalized_com"][:n, :, :])
        u10 = u10.reshape(u10.shape + (1,))
        v10 = np.array(ds["v10_normalized_com"][:n, :, :])
        v10 = v10.reshape(v10.shape + (1,))

        # 合并两个数组
        wind_true_phy = np.concatenate((u10, v10), axis=-1)

        msl = np.array(ds["msl_normalized_com"][:n, :, :])
        msl = msl.reshape(msl.shape + (1,))

    num_scenes = v10.shape[0]

    # 清理内存
    gc.collect()

    num_train = int(round(num_scenes*(1.0-frac_validate)))

    phy_scenes = {
        "train": (
            wind_true_phy[:num_train,...],
            msl[:num_train,...]
        ),
        "validate": (
            wind_true_phy[num_train:,...],
            msl[num_train:,...]
        )
    }
    # (8760,64,64,1)
    return phy_scenes




 # 重缩放场景数据，从(-1, 1)范围转换到(Z_range[0], Z_range[1])范围
def rescale_scene(scene, Z_range=(-40,40), missing_max=-99):
    sc = Z_range[0] + (scene+1)/2.0 * (Z_range[1]-Z_range[0])
    # 将小于等于missing_max的值设置为NaN，表示缺失数据
    # sc[sc <= missing_max] = np.nan
    return sc

# 该函数用于生成批量场景训练数据，每次迭代返回指定大小的批量数据。
def gen_batch(cs_scenes, modis_vars, modis_mask, v10, msl, batch_size):
    ind = np.arange(cs_scenes.shape[0], dtype=int)
    # np.random.shuffle(ind)
    while len(ind) >= batch_size:      
        idx = ind[:batch_size]
        ind = ind[batch_size:]
        yield (cs_scenes[idx,...], modis_vars[idx,...], modis_mask[idx,...], v10[idx,...], msl[idx,...])




def save_opt_weights(model, filepath):
    with h5py.File(filepath, 'w') as f:
        # Save optimizer weights.
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                # Default values of symbolic_weights is /variable for theano
                if K.backend() == 'theano':
                    if hasattr(w, 'name') and w.name != "/variable":
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                else:
                    if hasattr(w, 'name') and w.name:
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val
            print(f"Saved optimizer weights: {[name.decode('utf8') for name in weight_names]}")

def load_opt_weights(model, filepath):
    try:
        with h5py.File(filepath, mode='r') as f:
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n if isinstance(n, str) else n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            # optimizer_weight_names = [n.decode('utf8') for n in
            #                           optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            print(f"Loaded optimizer weights: {optimizer_weight_names}")
            model.optimizer.set_weights(optimizer_weight_values)

    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"Error loading optimizer weights: {e}")

