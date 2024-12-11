# cs224w

Processing dataset from Structural Antibody Database (SAb-Dab)
https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab

Custom code to get the data to include PDBID, antibody-seq, cdr-seq (as an antibody sequence mask for H1, H2, and H3), antibody-coords, antibody-atype, antigen-coords, antigen-atype.

Only antibodies with known antigen-bound structures were chosen.

The antibody-cdr sequence was deteremined by masking indices in the implementation described by Wegong Jin et al., but this processing method searches the subsequence instead to enusre that the CDR regions don't suffer from amino acid frameshifts.
