# [modsem](https://kss2k.github.io/intro_modsem/) <img src="man/figures/modsem.png" alt="Logo" align = "right" height="139" class="logo">
This is a version of the `modsem` package where the `qml` is parallelized using `RcppParallel`.
If you're using a large dataset, it might be faster to use this version, but for smaller datasets, 
the original version is likely faster.



# To Install 
```
devtools::install_github("kss2k/modsem", build_vignettes = TRUE, 
                         ref = "parallelized")
```

# Methods/Approaches

There are a number of approaches for estimating interaction effects in SEM. In `modsem()`, the `method = "method"` argument allows you to choose which to use.

- `"ca"` = constrained approach (Algina & Moulder, 2001)
    - Note that constraints can become quite complicated for complex models, 
      particularly when there is an interaction including enodgenous variables.
      The method can therefore be quite slow. 
- `"uca"` = unconstrained approach (Marsh, 2004)
- `"rca"` = residual centering approach (Little et al., 2006)
- `"dblcent"` = double centering approach (Marsh., 2013)
  - default 
- `"pind"` = basic product indicator approach (not recommended)
- `"lms"` = The Latent Moderated Structural equations (LMS) approach, see the [vignette](https://kss2k.github.io/intro_modsem/articles/lms_qml.html)
- `"qml"` = The Quasi Maximum Likelihood (QML) approach, see the [vignette](https://kss2k.github.io/intro_modsem/articles/lms_qml.html)
- `"mplus"` 
  - estimates model through Mplus, if it is installed

# New Features version 1.0.1
- Interaction effects between endogenous and exogenous variables are now possible by default with QML-approach.
- Interaction effects between two endogenous variables are now possible with the LMS 
  and QML approach, using the 'cov.syntax' argument, see the [vignette](https://kss2k.github.io/intro_modsem/articles/interaction_two_etas.html)
  for more information.
- Improved `summary()` function for LMS and QML: 
    1. Standardized estimates are now available for the LMS and QML approach, 
    using the `standardized = TRUE` argument.
    2. The `summary()` function now also returns the RMSEA, Chi-Square, AIC, BIC, and Expected covariance matrix for the LMS and QML approach.
    3. The `summary()` function now resembles the output of the `summary()` function from the `lavaan` package.
- Added post-estimation functions for LMS and QML:
    1. `modsem_inspect()` for inspecting the results of the LMS and QML approach 
    2. `fit_modsem_da()` caluculates a variety of fit indices for the LMS and QML approach, 
        RMSEA, Chi-Square, AIC, BIC, and Expected covariance matrix.
    3. `vcov()` returns the variance-covariance matrix of the parameter estimates for the LMS and QML approach. 
    4. `coef()` returns the parameter estimates for the LMS and QML approach.

# Examples 

## One interaction
```
library(modsem)
m1 <- '
  # Outer Model
  X =~ x1 + x2 +x3
  Y =~ y1 + y2 + y3
  Z =~ z1 + z2 + z3
  
  # Inner model
  Y ~ X + Z + X:Z 
'

# Double centering approach
est1_dca <- modsem(m1, oneInt)
summary(est1_dca)

# Constrained approach
est1_ca <- modsem(m1, oneInt, method = "ca")
summary(est1_ca)

# QML approach 
est1_qml <- modsem(m1, oneInt, method = "qml")
summary(est1_qml, standardized = TRUE) 

# LMS approach 
est1_lms <- modsem(m1, oneInt, method = "lms") 
summary(est1_lms)
```

## Theory Of Planned Behavior
```
tpb <- "
# Outer Model (Based on Hagger et al., 2007)
  ATT =~ att1 + att2 + att3 + att4 + att5
  SN =~ sn1 + sn2
  PBC =~ pbc1 + pbc2 + pbc3
  INT =~ int1 + int2 + int3
  BEH =~ b1 + b2

# Inner Model (Based on Steinmetz et al., 2011)
  # Causal Relationsships
  INT ~ ATT + SN + PBC
  BEH ~ INT + PBC
  BEH ~ PBC:INT
"

# double centering approach
est_tpb_dca <- modsem(tpb, data = TPB, method = "dblcent")
summary(est_tpb_dca)

# Constrained approach using Wrigths path tracing rules for generating
# the appropriate constraints
est_tpb_ca <- modsem(tpb, data = TPB, method = "ca") 
summary(est_tpb_ca)

# LMS approach 
est_tpb_lms <- modsem(tpb, data = TPB, method = "lms")
summary(est_tpb_lms, standardized = TRUE) 

# QML approach 
est_tpb_qml <- modsem(tpb, data = TPB, method = "qml") 
summary(est_tpb_qml, standardized = TRUE)
```
## Interactions between two observed variables
```
est2 <- modsem('y1 ~ x1 + z1 + x1:z1', data = oneInt, method = "pind")
summary(est2)

## Interaction between an obsereved and a latent variable 
m3 <- '
  # Outer Model
  X =~ x1 + x2 +x3
  Y =~ y1 + y2 + y3
  
  # Inner model
  Y ~ X + z1 + X:z1 
'

est3 <- modsem(m3, oneInt, method = "pind")
summary(est3)
```
