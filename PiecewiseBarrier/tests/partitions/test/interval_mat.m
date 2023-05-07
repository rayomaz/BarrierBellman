% Plotting ibp bounds on erf functions

clc; clear; close;

filename = "linearsystem_2000.mat";

load(filename)

% Probability distribution on hyperspace
hypercubes = length(upper_partition);
prob_transition_lower = zeros(hypercubes, hypercubes);
prob_transition_upper = zeros(hypercubes, hypercubes);
prob_unsafe_upper = zeros(1, hypercubes);
prob_unsafe_lower = zeros(1, hypercubes);

for zz = 1:2

    for jj = 1:hypercubes 
        
        for ii = 1:hypercubes 
    
            if zz == 1
                p_ij = true;              % Transition Xj to Xi
            elseif zz == 2 
                p_ij = false;             % Transition Xj to Xs
            end
    
            refine = 100;
            x_space = linspace(lower_partition(jj), ...
                               upper_partition(jj), refine);
    
            % True erf
            prob_true = zeros(1, length(x_space));
            sigma = 0.1;
            epsilon = 1e-6;
            m = 1;      % sys dim
            const = 1/(2^m);
            
            if p_ij == false
                vl = min(lower_partition);
                vu = max(upper_partition);
            else
                vl = lower_partition(ii);
                vu = upper_partition(ii);
            end
    
            % Prod of erfs
            for pp = 1:length(x_space)
                y = 0.95*x_space(pp);
                erf_low = (y - vl)/(sigma*sqrt(2));
                erf_up = (y - vu)/(sigma*sqrt(2));
                if p_ij == false
                    prob_true(pp) = 1 - const*(erf(erf_low) - erf(erf_up));
                else
                    prob_true(pp) = const*(erf(erf_low) - erf(erf_up));
                end
                % Notice this computes Pu --> Pu = 1 - Ps
            end
    
            if p_ij == true
                prob_transition_upper(jj, ii) = max(prob_true);
                prob_transition_lower(jj, ii) = min(prob_true);
            elseif p_ij == false
                prob_unsafe_upper(jj) = max(prob_true);
                prob_unsafe_lower(jj) = min(prob_true);
            end
        
        end
    end
end


save(filename,"prob_transition_upper", ...
              "prob_transition_lower", ...
              "prob_unsafe_lower", ... 
              "prob_unsafe_upper", ...
              '-append')



