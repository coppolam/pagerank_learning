function [out, yes, states, reach, G] = reach_graph_direct(states, transitions, states_des, mode)
% reach_graph_direct checks that, given a set of states, transitions, and desired states, there exist direct transitions between the states and the desired states.
% Depending on the input mode, it can check:
% 'all': All desired states can be reached from all states
% 'any': At least one desired state can be reached from all states
%
% Mario Coppola, 2018

if numel(states) ~= numel(transitions)
    error('States and transitions should be the same size')
end

emptycells = cellfun(@isempty, transitions);

startingpoints = states(~emptycells);
transitions = transitions(~emptycells);
cellsize = cell2mat(cellfun(@length, transitions, 'uni', false));
allstartingpoints = rude(cellsize, startingpoints);
alltransitions = cat(2, transitions{:});

% Generate graph
G = digraph(double(allstartingpoints), double(alltransitions));
reach = cell(1, numel(states));
a = ones(1, numel(states_des));
states = unique(allstartingpoints);

yes = zeros(1, numel(states));

% For each state, we analyze the successors and see if they are a desired state
for i = 1:numel(states)
    reach{i} = unique(successors(G, states(i)));
    if strcmp(mode, 'all')
        yes(i) = all(ismember(states_des, reach{i}));
        a = and(a, ismember(states_des, reach{i}));
    elseif strcmp(mode, 'any')
        yes(i) = any(ismember(states_des, reach{i}));
        a = or(a, ismember(states_des, reach{i}));
    end
end

if all(yes) && strcmp(mode, 'all') 
    % All desired states can be reached from all states
    out = 1;
elseif any(yes) && strcmp(mode, 'any')
    % A desired state can be reached from all states
    out = 1;
else
    out = 0;
end

end
