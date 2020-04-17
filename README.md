[francois chollet implementation of deep dream in keras (with tf1.x):](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.2-deep-dream.ipynb)

[another basic example using tf2:](https://github.com/goyetc/deep_dreams/blob/master/CG2970_deep_dream_basic.ipynb)

fchollet notebook [augmented](view/8.2-deep-dream-augmented.ipynb) and adapted into a [batch run](view/auto-dream.py)

[Launcher](endpoints/launchers#/launchers/5e5c1eabc9e77c00070757c5/auto-deep-dream) - uses a static set of options
- layers to lift from pretrained model
- relative weight of each layer during gradient ascent
- 'octaves', or number of scales at which to process the image and successively re-apply base details to maintain resolution through the process
- octave scaling ratio
- limits on iterations and associated loss per octave

Images around 1MB work best; too large and the effect of this implementation is diminished; also, cannot display large images in-line in results. 

To Do:
- save the pre-trained model to a Domino Dataset and load it at runtime so it doesn't have to be downloaded each run
- set up inputs so simple comparison of original and augmented image can be done in results
- create second launcher that exposes hyperparameters discussed above as inputs
- experiment with tiling and other methods to improve aesthetic and depth of outputs
  - 
  
Example output: 
![image](raw/latest/final_dream.png)
![image](raw/latest/results/20190804194001_IMG_4194final_dream.png)