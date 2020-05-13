# Instantiation dataset, modified #
This folder contains a version of the Instantiation dataset as we have modified it for our purposes. 
For details about the definition of the various variants (the different files), see the paper.
The files have tab-separated columns.
To see how we read the files, see for instance `read_dataset()` in `paper_results.py`.

**Original dataset**:

http://www.ims.uni-stuttgart.de/data/Instantiation.html

From Boleda, Gupta and Pad\'o 2017, Instances and concepts in distributional space. Proceedings of EACL.

**Our modifications:**

- Restricted to the categories with at least 5 entities.
- A more challenging NotInst variant, where the confounder caregories are taken from the same ontological domain.  
- Test partition split into 83 folds, for proper prevention of memorization effects.
