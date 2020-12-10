%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function assignParams(opts,params)
%
%   Assigns all of the fields of the struct params to variables in the
% caller's workspace. Used for overwriting defaults with parameter inputs.
% Note that it expects 'opts', which is a list of possible variables. I
% typically define default values for all parameters, then call this with
% "who" as the input. That way I only overwrite parameters that a function
% actually needs, even if params has many more fields.
%
% INPUTS:
%   opts   : possible variables to overwrite (typically "who")
%   params : parameter struct with fields whose names are variables
%
% EXAMPLES:
%   e.g. to overwrite default value of variable foo
%       foo = 'bar';
%       params = struct('foo','d');
%       assignParams(who,params);
%
% Written by Matt Perich. Updated Feb 2017.
% 
% Inspired by assignopts by Maneesh Sahani. Credit where credit is due, ya know?
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function assignParams(opts,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isstruct(params), error('Params is not a struct.'); end
if length(params) > 1, error('Params has multiple entries. Often this is caused by neglecting to wrap cell arrays in extra {} when defining a struct().'); end

% get all parameters
vars = fieldnames(params);

for var = 1:length(vars)
    % make sure this variable is an option
    if ismember(vars{var},opts)
        assignin('caller', vars{var}, params.(vars{var}));
    end
end