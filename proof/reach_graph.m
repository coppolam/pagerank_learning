function [out] = reach_graph(G, states_inquired, states_des)
%reach_graph Determines whether the desired state can all be reached from all states by local transitions.
%
% Mario Coppola, 2018

reach = cell(1, numel(states_inquired));
a = ones(1, numel(states_des));
states_inquired = unique(states_inquired);

out = 0; % Negative assumption. Some local states cannot be reached from all states

% If the graph is not connected then it's not ok.
if ~conncomp(G, 'Type', 'weak')
    disp("Graph is not connected")
    return;
end

% If the graph has less nodes than inquired it's not ok.
if size(G.Nodes) < numel(states_inquired)
    return;
end

yes = zeros(1, numel(states_inquired));
for i = 1:numel(states_inquired)
    reach{i} = unique(bfsearch(G, states_inquired(i)));
    yes(i) = all(ismember(states_des, reach{i}));
    a = and(a, ismember(states_des, reach{i}));
end

if all(yes) % All desired states can be reached from all states
    out = 1;
    disp("All final states can be reached from initial states")
end

end
