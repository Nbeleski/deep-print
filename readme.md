# DeepPrint Personal Implementation

This repository is a **personal, work-in-progress project** that implements the key ideas from the paper [_"Learning a Fixed-Length Fingerprint Representation" (DeepPrint) by Engelsma, Cao, and Jain_](https://arxiv.org/abs/1909.09901).

## Introduction

I started this project because I think DeepPrint is one of the more interesting approaches for learning compact, fixed-length representations of fingerprints. I believe the paper multi-task approach is really sound and it is a worthwhile exercise to try and achieve similar results. The original paper integrates domain knowledge—like automatic alignment and minutiae detection—directly into a deep network to create efficient, discriminative fingerprint embeddings. I highly recommend reading the paper for the technical details and motivations behind this method.

> **Note:** This implementation is not official, and is under active development. There will be missing features, bugs, and rough edges as I continue to work on it.

## Project Structure

- **`model.py`**  
  Main model definition. This file implements the core DeepPrint network architecture, including the alignment module, multi-branch feature extraction, and minutiae map prediction. As far as I understand, the implementation is as faithful as I could get from the paper description. I have made some choices such changing the minutiae map size to better match the respective head of the model, it shouldn't change much as far as I know.

- **`inception_v4.py`**  
  Implementation of the custom Inception-v4 backbone and related building blocks used within the DeepPrint architecture.

- **Training Notebook**  
  The training notebook (see the relevant `.ipynb` file in the repo) contains experiments and training code for testing this implementation on fingerprint datasets.

## About

There is a lot going on to train this model, this is a working in progress and I am still figuring out the details. For example, the affine transformation application to the map is not trivial, the aligment model can get stuck on bad values, etc.

Future version should have cleaner code and more comments.

## Status

- This project is a work in progress.
- Major features and experimental results are still under development.
- Not production-ready or meant for deployment.

## Reference

If you're interested in the original method, see:  
**Engelsma, J.J., Cao, K., & Jain, A.K. (2019). Learning a Fixed-Length Fingerprint Representation. [arXiv:1909.09901](https://arxiv.org/abs/1909.09901)**

