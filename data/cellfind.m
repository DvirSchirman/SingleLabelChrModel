function inds = cellfind (cell_in,query)

inds = find(cell2mat(cellfun(@(x) strcmp(x,query),cell_in,'un',0)));