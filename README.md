# HiCComp: Multiple-level Comparative Analysis of Hi-C Data by Triplet Network
## Contact
zhang348 at email.sc.edu

## Introduction
Hi-C technique is an important tool for the study of 3D genome organization. In the past few years, we have seen an explosion of Hi-C data in a variety of cell/tissue types. While these publicly available data presents an unprecedented opportunity to interrogate chromosomal architecture, how to quantitatively compare Hi-C data from different tissues and identify tissue-specific chromatin interactions remains challenging. 

Here, we present HiCComp, a comprehensive framework for comparing Hi-C data. HiCComp utilizes convolutional neural networks to extract key features in Hi-C interaction matrices in a fully automatic way. The core component of HiCComp is a triplet network, which contains three identical convolutional neural networks with shared parameters. The inputs to our network are three Hi-C matrices: two of them are biological replicates from the same cell type and the third one is from another cell type. 

The HiCComp network takes advantages of the two biological replicates to estimate the natural variation in the experiments and further use it to identify significant variations between Hi-C matrices from different cell types. Furthermore, we incorporate systematic occluding method into our framework so that we can identify the dynamic interaction regions from Hi-C maps. Finally, we show that the dynamic regions between two cell types are enriched for transcription factor binding sites and histone modifications that are associated with cis-regulatory functions, suggesting these variations in 3D genome structure are potentially gene regulatory events.

