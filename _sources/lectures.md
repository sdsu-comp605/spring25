# Lectures

The class material may include Jupyter-based notebooks.

You can view them here, or open them to interact. You can use any environment for your local development environment, or use the SDSU's [JupyterHub](https://jupyterhub.sdsu.edu/) on the [Instructional Cluster](https://sdsu-research-ci.github.io/instructionalcluster) to experiment and develop without a local install. If you have never logged-in before, check SDSU's Research & Cyberinfrastructure [resources for students](https://sdsu-research-ci.github.io/instructionalcluster/students). Once you are on the [Instructional Cluster](https://sdsu-research-ci.github.io/instructionalcluster) page, simply click on the `Launch JupyterHub` button.

If you decide to use SDSU's [Instructional Cluster](https://sdsu-research-ci.github.io/instructionalcluster)'s [JupyterHub](https://jupyterhub.sdsu.edu/), you can download each individual notebook from the class website, under the [lectures](https://sdsu-comp605.github.io/spring25/lectures.html) section on your machine, and once opened Jupyter on SDSU's JupyterHub you can select `Upload Files` and upload the desired notebook in the web app to interact with.

## Environment

This explains how I configure my environment so you can experiment with the lectures locally, or adapt these tools for your own use.

### Install Required Packages

First of all, for Jupyter Notebooks involving the Julia programming language, download and install [Julia](https://julialang.org/downloads/).

To interact with the notebooks, you want to install JupyterLab (new, richer ecosystem), via

```
pip install jupyterlab
```

Once installed, launch JupyterLab with:

```
jupyter lab
```

and selcet the Notebook app.

You can also only install the classic Jupyter Notebook standalone app via

```
pip install notebook
```

and to run the notebook type:

```
jupyter notebook
```

And for these (using Julia in Jupyter) you can run

```
julia -e 'import Pkg; Pkg.add("IJulia")'
```

from your terminal, or first start a Julia session with
```
julia
```

and then type:
```julia
]add IJulia
```

go back to your Julia REPL using backspace and type

```julia
julia> using IJulia
```

and then
```julia
julia> notebook()
```

to run the notebook.

### Best practices

* When saving, use `Kernel -> Restart & Clear Output` to keep the stored data and diffs in the notebook small.
* For your images, several formats will do. Just keep in mind that PDF files are not web-friendly. PDF images can be converted to SVG using `pdf2svg`, and the result will still look sharp (unlike PNG) no matter the scale or zoom.

## Online resources:

- For the first part of our course, an online MOOC: [LAFF-On Programming for High Performance (LAFF-On-PfHP)](https://www.cs.utexas.edu/users/flame/laff/pfhp/LAFF-On-PfHP.html) will be particularly useful.

I encourage you to watch the videos before class. When watching the videos, really you should be watching for the _big picture_ and we will revisit most of the material in lecture as well, mainly you will have a better learning experience if the in class lectures are the first time you've seen some of the material.

## Acknowledgements
For most of this course's materials, I owe huge thanks to my postdoc mentor, [Jed Brown](https://jedbrown.org), Professor at CU Boulder, and former colleague, Dr. Jeremy Kozdon.
