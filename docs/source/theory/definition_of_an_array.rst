==========================
The Definition of an Array 
==========================

Since we use the notion of an "n-dimensional array" so frequently, we 
propose a model below that we will use throughout the discussions below.

Loosely speaking, an n-dimensional array is a labelling of the vertices of
a "Cartesian" graph by some data type. 

.. admonition:: Definition

   definition of **cartesian graph**

.. admonition:: Definition

   an **array** is a map:

   where the target is the **data type** of the array.

   or equivalently a point in this mapping space: 

.. admonition:: Definition

   the dimension of the array
   the shape of the array 

.. admonition:: Definition

   array is **parametrized** by the graph

.. note:: 

   **array** is a labelling of the vertices of a Cartesian graph by number

.. admonition:: Definition

   **riemannian array** is a labelling of the edges by positive numbers.

.. note:: 

   certain configurations of points in the I^n give every array a Riemannian 
   structure. This corresponds to the `series` class/parameter in Pandas.

Application: Slicing
-------------------

The perspective of an array as a map has a profound upside: slicing are representble.
In other words, slicing operations can be computed as pulling back/piping into
explicit maps of graphs. 

Appendix: Core Categories
-------------------------

Precise articulation of the structure of ML objects is challenging.
A categorical approach to spectral graph theory will be my primary tool
to address these challenges.

First, I will define some geometric and algebraic categories. In general, our
ML objects will be sheaves over these geometric categories with values in
our algebraic categories (and sections thereover).

.. admonition:: Definition of Primary Geometric category

   Category of graphs

.. admonition:: Definition: Primary Algebraic Category 

   Category of vector spaces.

.. warning::

   We actually mean "the" :math:`\infty`-category of "vector spaces." 
   This :math:`\infty`-category admits many well-behaved presentation 
   as model categories. In other words, computations can be done in a 
   strict context. 

.. note::

   This category is a shadow of the category we're actually interested
   in: a canonically smooth stratified spaces (that appropriately regular). 
   However, there are many instances in which Poincare duality allows one
   to get away with a purely graphical description.

Sheaves pipe the geometric into the algebraic.

.. admonition:: Example

   The constant R-valued sheaf. 

.. admonition:: Definition: Geometric Structures

   a weight structure

.. note:: 

   Ingredient constructions for energy 

.. admonition:: Definition: 

   The energy of a section.
.. note::

   Difference between xarray and numpy: x-array datasets are parametrized 
   by stratified shapes with geometric structure. 

.. admonition:: Definition

   **Harmonic/Low Energy Approximations** is defined 

