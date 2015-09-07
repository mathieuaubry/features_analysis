[Understanding deep features with computer-generated imagery](http://imagine.enpc.fr/~aubrym/projects/features_analysis/) source code
===========

Here you will find a Matlab implementation of the algorithm described
in the following paper:

   Mathieu Aubry and Bryan C. Russell.
   Understanding deep features with computer-generated imagery.
   ICCV 2015.
   [PDF](http://imagine.enpc.fr/~aubrym/projects/features_analysis/texts/understanding_deep_features_with_CG.pdf) | [BibTeX](http://imagine.enpc.fr/~aubrym/projects/features_analysis/texts/2015-understanding-deep-features_bibtex.html) | [Project page](http://imagine.enpc.fr/~aubrym/projects/features_analysis/)

Note that this implementation has minor differences with the one used to generate the results shown in the paper.

For any questions or feedback regarding the source code please contact [Mathieu Aubry](mailto:mathieu.aubry@imagine.enpc.fr). 


### DOWNLOADING THE CODE:

You can download a [zip file of the source code](https://github.com/mathieuaubry/features_analysis/archive/master.zip) directly.  

Alternatively, you can clone it from GitHub as follows:

``` sh
$ git clone https://github.com/mathieuaubry/features_analysis.git
```

### DOWNLOADING THE DATA:


To run the part of the code using CAD models, you will need to download the rendered views of the chair CAD
models. Depending on the experiment you want to run, you can download the [rotated CAD models](http://imagine.enpc.fr/~aubrym/projects/features_analysis/data/render_rotation.zip) or the front-facing models with varying [foreground color](http://imagine.enpc.fr/~aubrym/projects/features_analysis/data/render_fg.zip), [background color](http://imagine.enpc.fr/~aubrym/projects/features_analysis/data/render_bg.zip) or [light orientation](http://imagine.enpc.fr/~aubrym/projects/features_analysis/data/render_light.zip).

Alternatively, you can use your own rendered 3D models. We plan to release our own renderer.

### DEPENDENCIES:

This code uses [Caffe](http://caffe.berkeleyvision.org/) and it's Python interface to compute CNN features. You can install it following the instructions from the [caffe installation webpage](http://caffe.berkeleyvision.org/installation.html)

### RUNNING THE CODE:

The steps of running the code are described in the "Perform Analysis" iPython notebook 


### ACKNOWLEDGMENTS:

We used a subset of the 3D models of [ModelNet](http://modelnet.cs.princeton.edu/)


