# Use this script to check the versions of the packages you installed

import numpy
import pandas
import sklearn
import textblob

# The versions listed above are pinned in the requirements file
print(f"numpy version: {numpy.__version__}")  # Should be 1.25.0
print(f"pandas version: {pandas.__version__}")  # 2.0.3
print(f"scikit-learn version: {sklearn.__version__}")  # 1.2.2
print(f"textblob_version: {textblob.__version__}")  # 0.17.1
