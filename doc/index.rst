.. popsregression documentation master file

:notoc:

################
POPS Regression
################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/tomswinburne/POPS-Regression>`__ |
`Issues & Ideas <https://github.com/tomswinburne/POPS-Regression/issues>`__

Bayesian regression for low-noise data with misspecification uncertainty
estimation using the POPS (Pointwise Optimal Parameter Sets) algorithm.

``popsregression`` provides :class:`~popsregression.POPSRegression`, a
scikit-learn compatible estimator that extends
:class:`~sklearn.linear_model.BayesianRidge` to estimate weight uncertainties
accounting for model misspecification. This is particularly useful for
surrogate models fit to near-deterministic data, where standard Bayesian
regression significantly underestimates predictive uncertainty.


.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/img/index_getting_started.svg
        :class-card: intro-card
        :shadow: md

        Installation and quick introduction to POPS Regression.

        +++

        .. button-ref:: quick_start
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the getting started guideline

    .. grid-item-card::  User guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        Background on the POPS algorithm and how to use it effectively.

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        Detailed API documentation for POPSRegression.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Examples
        :img-top: _static/img/index_examples.svg
        :class-card: intro-card
        :shadow: md

        Gallery of examples demonstrating POPS Regression usage.

        +++

        .. button-ref:: general_examples
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the gallery of examples


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quick_start
    user_guide
    api
    auto_examples/index
