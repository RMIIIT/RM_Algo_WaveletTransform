
import numpy as np
from PIL import Image
import os

'''
The Haar Wavelet  transform  used in lossy image compression.

'''

''''Main function'''

scale = np.sqrt(2.)

def main():
  srcImg = to_float(load('image.png')) #loading image

  coeffs = haar_2d(srcImg)
  dominant_coeffs = keep_ratio(coeffs, .05)
  lossy = ihaar_2d(dominant_coeffs)

  save('harr-coeff.png', bipolar(coeffs))
  save('dominant-coeff.png', bipolar(dominant_coeffs))
  save('output.png', from_float(lossy))

#  haar wavelet transformation related code:



def haar_wavelet_tranform(data):
  if (len(data) == 1):
    return data.copy()
  assert len(data) % 2 == 0   , "length needs to be even"
  mid_val = (data[0::2] + data[1::2]) / scale
  side_val = (data[0::2] - data[1::2]) / scale
  return np.hstack((haar_wavelet_tranform(mid_val), side_val))

def ihaar(data):
  if len(data) == 1:
    return data.copy()
  assert len(data) % 2 == 0, "length needs to be even"
  mid = ihaar(data[0:int(len(data)/2)]) * scale
  side = data[int(len(data)/2):] * scale
  out = np.zeros(len(data), dtype=float)
  out[0::2] = (mid + side) / 2.
  out[1::2] = (mid - side) / 2.
  return out

def haar_2d(srcImg):
  h,w = srcImg.shape
  rows = np.zeros(srcImg.shape, dtype=float)
  for y in range(h):
    rows[y] = haar_wavelet_tranform(srcImg[y])
  cols = np.zeros(srcImg.shape, dtype=float)
  for x in range(w):
    cols[:,x] = haar_wavelet_tranform(rows[:,x])
  return cols

def ihaar_2d(coeffs):
  h,w = coeffs.shape
  cols = np.zeros(coeffs.shape, dtype=float)
  for x in range(w):
    cols[:,x] = ihaar(coeffs[:,x])
  rows = np.zeros(coeffs.shape, dtype=float)
  for y in range(h):
    rows[y] = ihaar(cols[y])
  return rows

def keep_ratio(data, ratio):
  """
  Keep only the strongest values.
  """
  magnitude = sorted(np.abs(data.flatten()))
  idx = int((len(magnitude) - 1) * (1. - ratio))
  return np.where(np.abs(data) > magnitude[idx], data, 0)

# --- graphics-related code:

def to_float(srcImg, gamma=2.2):
  """
  Convert uint8 image to linear floating point values.
  """
  return np.power(srcImg.astype(float) / 255, gamma)

def from_float(srcImg, gamma=2.2):
  """
  Convert from floating point, doing gamma conversion and 0,255 saturation,
  into a byte array.
  """
  out = np.power(srcImg.astype(float), 1.0 / gamma)
  out = np.round(out * 255).clip(0, 255)
  return out.astype(np.uint8)

def bipolar(srcImg):
  """
  Negative values are red, positive blue, and zero is black.
  """
  h,w = srcImg.shape
  srcImg = srcImg.copy()
  srcImg /= np.abs(srcImg).max()
  out = np.zeros((h, w, 3), dtype=float)
  a = .005
  b = 1. - a
  c = .5
  out[:,:,0] = np.where(srcImg < 0, a + b * np.power(srcImg / (srcImg.min() - 0.001), c), 0)
  out[:,:,2] = np.where(srcImg > 0, a + b * np.power(srcImg / (srcImg.max() + 0.001), c), 0)
  return from_float(out)

def load(fileName):
  return np.asarray(Image.open(fileName).convert(mode='L'))

def save(fileName, srcImg):
  assert srcImg.dtype == np.uint8
  Image.fromarray(srcImg).save(fileName)
  print ('wrote', fileName)


def delImages():
  if os.path.exists("harr-coeff.png"):
    os.remove("harr-coeff.png")
  if os.path.exists("dominant-coeff.png"):
    os.remove("dominant-coeff.png")
  if os.path.exists("output.png"):
    os.remove("output.png")

def test_func(val):
  if(val ==1):
    main()
  else:
    delImages()


if __name__ == '__main__':
  val=1
  test_func(val)

