## Papers to read regarding fourier domain transformers

### Scattering Vision Transformer: Spectral Mixing Matters
- https://papers.neurips.cc/paper_files/paper/2023/file/a97f8072e51a785434b2da3e9cbf5aae-Paper-Conference.pdf

    It is highly improbable that the level of complexity in this work is merited by its very marginal improvement over prior work.  The presentation is rife with errors and it seems unlikely that this direction will be profitable.  Anyway, it isn't based on fourier or wavelet or shearlet analysis.  It has good references for us though.

- https://papers.neurips.cc/paper_files/paper/2022/file/5a8177df23bdcc15a02a6739f5b9dd4a-Paper-Conference.pdf
- https://arxiv.org/abs/2104.02555
- https://ieeexplore.ieee.org/document/9834158
- https://openaccess.thecvf.com/content/CVPR2022W/CVMI/papers/Buchholz_Fourier_Image_Transformer_CVPRW_2022_paper.pdf
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840001.pdf
- http://ecmlpkdd2017.ijs.si/papers/paperID11.pdf

## Fourier Transformers
- https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_An_Image_Patch_Is_a_Wave_Phase-Aware_Vision_MLP_CVPR_2022_paper.pdf
- https://arxiv.org/abs/2105.03824
- https://arxiv.org/abs/2107.00645
- https://arxiv.org/abs/2111.13587

## Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning
- https://arxiv.org/abs/2207.04978



### Wavelet-Based Image Tokenizer for Vision Transformers
- https://arxiv.org/abs/2405.18616

    This paper utilizes the approximate structure of the sparsity in the wavelet analysis of images to reduce the size of the tokens input to vision transformers.  Because each step of the analysis reduces the resolution exponentially, but increases the size of their tokens linearly, then by introducing structured sparsity into the different scales of the analysis the size of the resulting embedding can be reduced.  This is tantamount in reality to filtering out high-frequency components (though their framework allows for the reverse to occur).  Either way, a heuristic decision is made based on the relative importance of different frequencies in the input which is determined by the rate of occurence of that "frequency" or wavelet response.  If the low frequency elements occur with greater magnitude coefficients more frequently then these elements are preserved.  There are several weaknesses to this approach:

    1. If the frequency elements that are important for the task at hand appear with high magnitude seldom, then they are intentionally filtered out of the input which could lead to problems in the presence of class imbalance or similar

    2. The approach actually increases the size of the input by at least 300%, so sparsity of at least 75% is required to reduce the computational load at all.  I think I did the calc right on this one but some sparsity is definitely required and we do not get the benefits of arithmetic coding so our sparsity must be structured and lossy with high probability.

    seems easy enough to make these into shearlets: 
    - https://en.wikipedia.org/wiki/Daubechies_wavelet
    - https://en.wikipedia.org/wiki/Coiflet

    though coshrem must be popular for a reason...

    Remember: This paper mentions non-uniform gridding in the context of token merging, dropping, or similar as it relates to numerical differential equations.  This could be an important future direction.

### Fourier Neural Operator for Parametric Partial Differential Equations
https://arxiv.org/abs/2010.08895
