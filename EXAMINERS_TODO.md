- the abstract could represent better the content of the thesis
- ~~the proof of proposition 2.2.1 uses a discrete eigenbasis of the
operator, this could be included in the formulation of the proposition~~
  - &rightarrow; proposition formulation edited
- ~~it is unclear if the example on p.27 has been developed by the author
or is also part of the review, this should be underlined better~~
  - &rightarrow; added a reference to John Preskill's notes on the first paragraph of the Subsection,
    thus makiing it clear that the exmaples are part of a review of existing literature
- ~~Figure 3.2 (b), is there a specific motivation to look at the
detection probability in the frequency domain?~~
  - &rightarrow; added an extra Note to the Figure caption to answer the question
- ~~chapter 4: IMPORTANT: the operator Omega is not consistently defined
in the whole chapter, sometimes H_T = - hbar Omega, sometimes H_T= hbar
Omega, it is essential that the author chooses one definition and use
that definition consistently during the whole chapter 4~~
  - &rightarrow; I adopted the sign convention of arXiv:1504.04215. With this choice,
    I found only one place with inconsistent sign, on p. 51, at the beginning of
    the last paragraph (inline formula), now fixed.
    If there are any other inconsistencies, which I may have overlooked, please let  me know.
- ~~IMPORTANT: as the definition of Omega is not consistent in chapter 4,
it is also not clear if the definition of the Fourier transformation and
the inverse Fourier transformation is consistent and correctly used in
Chapter 4; the author should check the consistent definition and correct
use of the Fourier transformations in Chapter 4~~
  - &rightarrow; Once removed the occurrence of H_t with the "wrong sign", the Fourier transform and inverse should be used correctly.
      With the exception of eq. (4.19) and the Fourier transform just before, now fixed.
      Now, to summarize everything related to sign conventions (in pseudocode):
      - H_T = \hbar\Omega = -i\hbar (d/dt)
      - H_T is in analogy with linear momentum p = -i\hbar (d/dx)
      - The Fourier transform has minus sign in the exponent in the integrand,
          transforms time (position) representation into frequency (momentum) representation
      - The Inverse Fourier has plus sign in the exponent in the integrand,
          transforms frequency (momentum) representation into time (position) representation
      - "up to a Planck constant", we use Fourier transformation and its inverse for time and frequency
          in the same formal way they are used (in almost every textbook) for position and momentum
          in standard quantum mechanics
- ~~the calculation shown in eq. (4.41) is not correct and should be
corrected, in line with this, eqs. (4.42) and (4.43) should be better
discussed in the text~~
  - &rightarrow; To my understanding, there where two intermediate passages in that calculation,
      one of which redundant and the other incorrect. By removing them, the whole calculation in (4.41)
      appears now correct and hopefully clearer, together with (4.42) and (4.43).
      Eq. (4.42) is mostly a copy-paste of (the correct part of) of (4.41) with a minor rearrangement of factors at the end.
- ~~p.68: the author should explain in the text why the 40th and 41th
eigenvectors are used here and not some others, is there no eigenvector
with eigenvalue 0 is this case?~~
  - Reference to the 40th eigenvector has been removed now, as it is not mentioned any further, in fact.
  - As per eigenvector 41th, the text already explains "In order to illustrate interesting examples, not correspondin
      to the trivial
      evolution of an eigenstate of HS , but to some kind of Rabi oscillation or Larmor
      precession (...), we pick ... *just as an example* ..."
  - However, another paragraph has ben added, hopefully adding some more clarity.
- ~~Figure 4.1: the figure caption is unclear and need to be improved~~
  - Done, hopefully to satisfaction.
- ~~the axis labels of Figs. 4.1, 4.2 are too small~~
  - &rightarrow; Axis labels increased in size.
- eq. (4.60) is unclear and should be motivated better
- it should be explained why there is a difference between the numerical
numbers in eqs. (4.69) and (4.70)
- Figure 4.5, 4.7: the captions need to be extended

There are also minor typos that will be discussed by the internal examiner and the student.
