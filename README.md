# Grounded Image Situation Recognition
The repo builds upon the work [Collaborative Transformers for Grounded Situation Recognition](https://arxiv.org/abs/2203.16518) with following contributions:
- Improved the results using CLIP[2] Embeddings (ViT-B/32 and ViT-L/14@336px)
- Investigated the performance of using Object features extracted using Faster-RCNN
- Investigated the performance of contextualized role-aggregated image features
  
## Problem
- Grounded Situation Recognition is the task of predicting action (verbs) and localized entities (nouns) associated with a semantic role from a given image
- Let V , R, and N denote the sets of verbs, roles, and nouns defined in the task, respectively. For each verb, v ∈ V , a set of semantic roles, denoted by Rv ⊂ R, is predefined as its frame by FrameNet. ​
- GSR aims to predict a verb v of an input image and assign a grounded noun to each role in Rv

## Acknowledgements
Our code is modified and adapted from this repository:
- [Collaborative Transformers for Grounded Situation Recognition](https://github.com/jhcho99/CoFormer)          

