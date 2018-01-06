
function write_out(output,i,name)%,dname, dataset)
output = output - min(output);
output = output/max(output);
%output = output + 1e-10*randn(length(output),1);  % Break ties at random
fname = [name '.output'];
save(fname,'output','-ascii');
% Either copy the evaluation script in the current directory or
% change the line below with the correct path
