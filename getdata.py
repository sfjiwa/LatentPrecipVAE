import numpy as np
import tensorflow as tf
import torch
import netCDF4
import os
def parse_dataset(example_proto, img_size_h, img_size_w, img_size_d, dim_anno1, 
    dim_anno2, dim_anno3, dtype_img=tf.float64):
    features = {
        'inputs': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'annotations': tf.io.FixedLenFeature(shape=[dim_anno1], dtype=tf.float32),
        'psl_mean_ens': tf.io.FixedLenFeature(shape=[dim_anno2], dtype=tf.float32),
        'temp_mean_ens': tf.io.FixedLenFeature(shape=[dim_anno3], dtype=tf.float32),
        'year': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'month': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'day': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    }

    parsed_features = tf.io.parse_single_example(example_proto, features=features)
    image = tf.io.decode_raw(parsed_features["inputs"], dtype_img)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [img_size_h, img_size_w, img_size_d])
    annotations = parsed_features["annotations"]
    psl = parsed_features["psl_mean_ens"]
    temp = parsed_features["temp_mean_ens"]
    year = parsed_features["year"]
    month = parsed_features["month"]
    day = parsed_features["day"]

    return image, annotations, psl, temp, year, month, day

def climate_dataset(filenames, height, width, depth, dim_anno1, 
    dim_anno2, dim_anno3, dtype):

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parse_dataset(x, height, width, depth, 
        dim_anno1, dim_anno2, dim_anno3, dtype_img=tf.float64))

    return dataset

def test2(directory):
    DANNO1 = 1000
    DANNO2 = 1
    DANNO3 = 1
    # image dimensions
    HEIGHT = 128
    WIDTH = 128
    DEPTH = 1
    # data type
    DTYPE = tf.float64

    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords' #retrived from the paper's github link
    fname = os.path.join(tfrecords_filename)
    ds = climate_dataset(fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds
# dimensionality of annotations

ds=test2('')
ds  # image, annotations, psl, temp, year, month, day
precip=[]
year=[]
month=[]
day=[]
for a, b, c, d, e, f, g, in ds:
  a=a.numpy()
  e=e.numpy()
  f=f.numpy()
  g=g.numpy()
  precip.append(a)
  year.append(e)
  month.append(f)
  day.append(g)
precip = np.array(precip)
precip_1959 = precip[419:509]
print(len(precip_1959))
precip_1959 = np.expand_dims(np.squeeze(precip_1959,axis=3),axis=1)
precip_1959 = torch.from_numpy(precip_1959)
torch.save(precip_1959,"precip.pt")
f = netCDF4.Dataset('pl_EUR-44_CCCma-CanESM2_historical_r1i1p1_CCCma-CanRCM4_r2_day_19560101-19601231.nc','r')
print(f)
print(f.variables.keys())
psl=f.variables['ps']

psl=np.array(psl)
np.save(file='1956_1960_psl_original.npy', arr=psl)
slp = np.load('1956_1960_psl_original.npy')
slp.shape #1956-1960
slp_1959 = slp[-396:-306]
slp_1959 = np.expand_dims(slp_1959, axis=1)
torch.save(slp_1959,"slp.pt")