addpath('npy-matlab/npy-matlab')
folder = 'repos/pagerank_learning/data/learning_data_pfsm_exploration_particle_oriented/analyze/';


Am = readNPY([folder,'A.npy']);
A = cell(8,1);
for i = 1:size(Am,1)
    A{i} = reshape(Am(i,:,:),size(Am,2),size(Am,3));
end
H = readNPY([folder,'H.npy']);
E = readNPY([folder,'E.npy']);
p0 = readNPY([folder,'p_0.npy']);
pn = readNPY([folder,'p_n.npy']);
%%
format short
% pn(2,:) = 0
b0 = zeros(size(H));
for i=1:size(A,1)
    b0 = b0 + A{i} .* p0(:,i);
end

b1 = zeros(size(H));
for i=1:size(A,1)
    b1 = b1 + A{i} .* pn(:,i);
end

H1 = b1'./b0' .* H;
H1(isnan(H1)) = 0;
G = digraph(H1);
plot(G);

des = [7,11,13,14]+1;

simplicial = [1,2,4,8]+1;

des
reach_graph(G,1:16,des);
simplicial
reach_graph(G,1:16,1:16);


%1 0 0 0 0
%2 0 0 0 1
%3 0 0 1 0
%4 0 0 1 1
%5 0 1 0 0
%6 0 1 0 1
%7 0 1 1 0
%8 0 1 1 1
%9 1 0 0 0 
