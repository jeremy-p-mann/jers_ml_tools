# Current Iteration

1. Make README better.
    1. License
    1. Example usage
    1. Contributions
1. Transformers clean up 
    1. add fit method to transformers docs
        1. bettier, cumlt, reshaper
    1. add underscor to attributes 
    1. add examples
1. add docs on expected outcomes of betti and grayscaler test
1. Draw example pipeline
1. Pepify tests/transformer

# Next Iteration

1. add integration test ensuring pipelines assemble in sensible order 

1. Add example notebook with MNIST example
    1. Assemble pipeline:
        1. Cumulants | lasso 

# Future Iteration

1. Make cumulants faster by applying standard formulae 
1. Increase possible color dimension
1. update module template
    1. 
1. Add euler characteristic transformer


Super Structure

1. Context/Motivation
1. Current State
1. Future directions

# Talk Outline

1. Context
    1. Active project
    1. Motivated by computer vision/ml project on colorectal cancer
        1. more sophisticated feature engineering with some advantages over DL
        1. can you show some of the ad hoc analyses which went into it.
    1. General classical sklearn flavored ML + fairly sophisticated 
       (interpretable) math
    1. Goals:
        1. Well documented
        1. Well tested
        1. Reusable
1. Go through directory structure
    1. Boilerplate stuff
        1. license 
    1. cat makefile
    1. Dependencies/Pipfiles  
    1. setup.py
1. Go through documentation 
    1. use ghpages website
    1. transformers docs
    1. hyperparameters docs
    1. mention further directions
1. Outline how you develop
    1. CDD/TDD
    1. Presentation will mirror this process, let me know if there's a 
       section I should skip
1. Go through test
    1. Run test
    1. context.py
    1. Grayscaler
    1. (time permitting) Bettier
1. demo with MNIST in ipython/jupyter
1. Go through source code 
    1. cumulants
    1. grayscaler
1. go through git graph
1. Checkout evaluator branch (time permitting)
1. Go through to do list


