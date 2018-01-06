%function byte_ranksvm()
% The function should be called with the path of the dataset from
% the Letor 3.0 distribution. For instance, if you are in
% Gov/QueryLevelNorm, type: letor_ranksvm('2003_hp_dataset')

global X Xt;
dataset = '.'
%mkdir(dataset,['ranksvm']); % Where the prdictions and metrics will be stored
dname = './';
%mkdir(dname,[dataset '/ranksvm']); % Where the prdictions and metrics will be stored

i = 1;
%for i=1:1, % Loop over the folds
    % Read the training and validation data
    %dname = ['./11_features1/'];
    [X, Y ] = read_letor([dname '/trainCV.txt']);
    [Xt,Yt] = read_letor([dname '/valiCV.txt']);
    
    % Generate the preference pairs; see ranksvm.m for the format of this matrix.
    A = generate_constraints(Y);
    for j=1:8 % Model selection
        opt.lin_cg=1;
        C = 10^(j-4)/size(A,1); % Dividing C by the number of pairs
        w(:,j) = ranksvm(X, A, C*ones(size(A,1),1),zeros(size(X,2),1),opt);
        %map(j) = compute_map(Xt*w(:,j),Yt); % MAP value on the validation set
        ndcg(j) = 0.5 * compute_ndcg(Xt*w(:,j),Yt,5) + 0.5 * compute_ndcg(Xt*w(:,j),Yt,10);
    end;
    %fprintf('C = %f, MAP = %f\n',[10.^[-2:5]; map])
    fprintf('C = %f, NDCG = %f\n',[10.^[-3:4]; ndcg])
    %[foo, j] = max(map); % Best MAP value
    [foo, j] = max(ndcg); % Best NDCG value
    %j = 5
    w = w(:,j);
    % Print predictions and compute the metrics.
    %write_out(X*w,i,'train')% ,dname ,dataset)
    %write_out(Xt*w,i,'vali')%,dname, dataset)
    [Xt,Yt] = read_letor([dname '/submitCV.txt']);
    write_out(Xt*w,i,'temp')%,dname, dataset)
    [Xt,Yt] = read_letor([dname '/testCV.txt']);
    write_out(Xt*w,i,'final')%,dname, dataset)
    %disp (dname)
    %end;

%system(['perl Eval-Score-3.0.pl ' dataset '/Fold' num2str(i) '/' name ...
%    '.txt ' fname ' ' fname '.metric 0']);




