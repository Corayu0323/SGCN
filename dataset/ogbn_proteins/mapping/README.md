# ogbn-proteins

### nodeidx2proteinid.csv.gz

Mapping from node index into the unique protein names in different species. 
The first part of the protein names are taxnomomy/species id: E. coli (txid: 511145), A. thaliana (txid: 3702), S. cerevisiae (txid: 4932), C. elegans (txid: 6239), D. melanogaster (txid: 7227), D. rerio (txid: 7955), H. sapiens (txid: 9606), M. musculus (txid: 10090).
The second part is the unique identifier of the protein withint the species.

### labelidx2GO.csv.gz

Mapping from label index to the GO annotations (protein functions).

### Edge features

The 8-dimensional edge features represents the following weights:

[homology_weight,
neighborhood_weight,
fusion_weight,
cooccurence_weight,
coexpression_weight,
experimental_weight,
database_weight,
textmining_weight] 

## Data references

* Szklarczyk, D., Gable, A.L., Lyon, D., Junge, A., Wyder, S., Huerta-Cepas, J., Simonovic, M., Doncheva, N.T., Morris, J.H., Bork, P. and Jensen, L.J., 2018. STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets. *Nucleic Acids Research*, 47(D1), pp.D607-D613.
* Ashburner, M., Ball, C.A., Blake, J.A., Botstein, D., Butler, H., Cherry, J.M., Davis, A.P., Dolinski, K., Dwight, S.S., Eppig, J.T. and Harris, M.A., 2000. Gene ontology: tool for the unification of biology. *Nature Genetics*, 25(1), p.25.