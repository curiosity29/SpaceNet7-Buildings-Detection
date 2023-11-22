import os
import glob

images = sorted(glob.glob(os.path.join("C:\Test\Github", r"scripts*")))
print("the path: \n")
print(images)
