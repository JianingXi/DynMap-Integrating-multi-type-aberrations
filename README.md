# DynMap-Integrating-multi-type-aberrations-
Integrating multi-type aberrations from DNA and RNA through dynamic mapping gene space for subtype-specific breast cancer driver discovery

![image](https://github.com/JianingXi/DynMap-Integrating-multi-type-aberrations-/blob/master/bin/splash.png)

Developer: Jianing Xi <xjn@gzhmu.edu.cn>

School of Biomedical Engineering,

Guangzhou Medical University, China

## Instructions to DynMap (version 1.0.0)

Requirement
------------------------
* Python 3.9 or later
* Python package: numpy, os, time, random, pickle, torch 

### Run DynMap

To apply DynMap, please simply run the python script file `./run.py` and the "splicing-and-fusing" process will automatically run on the Knowledge graph of breast cancer genomic mutations in `./Data_KG`.


The Pipeline of DynMap
------------------------

### 1. Downloading from UCSC Xena

The data of TCGA somatic mutations of breast cancers (BRCA) are downloaded from UCSC Xena [1], which are located at `https://xenabrowser.net/`,
* `./cBioPortal_RawData/blca_tcga_pub/data_mutations_extended.txt`.

### 2. Reformating into Knowledge Graph Data

* The mutation data are reformat into Knowledge Graph data as a series of triplets, according to the mutation types of genes in samples.

* Different types the occurrence of gene aberrations in DNA, including 3’ Flank, 3’ UTR, 5’ Flank, 5’ UTR, frame shift del, frame shift ins, IGR, in frame del, in frame ins, intron, missense mutation,
nonsense mutation, nonstop mutation, silent, splice region, splice site, and translation start site.

* These differentially expressed RNAs of genes are regarded as RNA alternations of genes.

### 3. Run DynMap

To apply DynMap, please simply run the python script file `./run.py` and the "splicing-and-fusing" process will automatically run on the Knowledge graph of breast cancer genomic mutations in `./Data_KG`.

### 4. Output Files

In file `./Output/' + args.dataset + '/'`, there are seven output files:

* `./Output/Loss_train.txt`: the training loss values across different iterations.

* `./Output/Loss_valid.txt`: the validation loss values across different iterations.

* `./Output/embedding_ent_final.txt`: the text file of the final embedding matrix of entities.

* `./Output/embedding_rel_final.txt`: the text file of the final embedding matrix of relations.

* `./Output/embedding_ent_proj_final.txt`: the text file of the final embedding matrix of entity projections.

* `./Output/embedding_rel_proj_final.txt`: the text file of the final embedding matrix of relation projections.

* `./Output/model_para_%s_%s.pkl`: the pickle file of 

By calculating embeddings of potential driver gene entity "Gene: ***", driver entity "Driver: somatic_driver", and relation "" through function "projection_DynMap_pytorch_samesize", we can obtain the loss scores of the gene to be a driver.

References
------------------------
[1] Goldman, M., Craft, B., Hastie, M., Repecka, K., McDade, F., Kamath, A., Banerjee, A., Luo, Y., Rogers, ˇ D., Brooks, A. N., et al. (2019). The ucsc xena platform for public and private cancer genomics data visualization and interpretation. biorxiv, page 326470
