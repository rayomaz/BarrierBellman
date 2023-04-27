% Plotting ibp bounds on erf functions

clc; clear; close;

load("linearsystem_5.mat")

% Probability distribution on hyperspace
figure
hold on
grid on

% Notice: this example is for transition from Xj to Xi
% where i = j = 1
hypercubes = length(upper_partition);
prob_unsafe_lower = zeros(1, hypercubes);
prob_unsafe_upper = zeros(1, hypercubes);
for jj = 1:hypercubes 
    
    colors = {'b', 'k', 'm', 'r', 'c'};
    for ii = 1%:length(upper_partition)

%         p_ij = true;              % Transition Xj to Xi
        p_ij = false;               % Transition Xj to Xs
    

        x_space = linspace(lower_partition(jj), upper_partition(jj), 1000);

        
        % Bounds
        if p_ij == true
            A_low = lower_probability_bounds_A(ii, jj, 1, :);
            b_low = lower_probability_bounds_b(ii, jj, :);
    
            A_up = upper_probability_bounds_A(ii, jj, 1, :);
            b_up = upper_probability_bounds_b(ii, jj, :);
    
            prob_bound_lower = A_low * x_space + b_low;
            prob_bound_upper = A_up * x_space + b_up;
        else
            A_low = lower_safe_set_prob_A_matrix(2*jj - 1);
            b_low = lower_safe_set_prob_b_vector(2*jj - 1);
    
            A_up = upper_safe_set_prob_A_matrix(2*jj - 1);
            b_up = upper_safe_set_prob_b_vector(2*jj - 1);
    
            prob_bound_lower = (A_low * x_space + b_low);
            prob_bound_upper = (A_up * x_space + b_up);
        end

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

        min_prob_exact = min(prob_true);
        max_prob_exact = max(prob_true);

        prob_unsafe_lower(jj) = min_prob_exact;
        prob_unsafe_upper(jj) = max_prob_exact;

        if p_ij == true
            plot(x_space, prob_bound_lower, "LineWidth", 3, 'color', "k")
            plot(x_space, prob_bound_upper, "LineWidth", 3, 'Color', "b")
        end

        plot(x_space, prob_true, "LineWidth", 3, 'Color', "r")

        plot(x_space, min_prob_exact*ones(1, length(x_space)), "LineWidth", 3, 'Color', "g")
        plot(x_space, max_prob_exact*ones(1, length(x_space)), "LineWidth", 3, 'Color', "g")
    
    end
end

% labels = {'\color{black} lower bound', '\color{blue} upper bound', ...
%     '\color{red} true', '\color{green} exact'};
% legend(labels, 'Location', 'NorthEast', 'FontSize', 8, ...
% 'TextColor', 'black');
% 




