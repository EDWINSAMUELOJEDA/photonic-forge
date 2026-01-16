import skimage
print(f"skimage version: {skimage.__version__}")
print(f"skimage path: {skimage.__file__}")

try:
    import skimage.measure
    print(f"skimage.measure: {skimage.measure}")
    print(f"dir(skimage.measure): {dir(skimage.measure)}")
except ImportError as e:
    print(f"Failed to import skimage.measure: {e}")
