%% File Info.
%{
    solve.m
    -------
    This code solves the model with UBI, linear taxes, and computes welfare.
%}

%% Solve class.
classdef solvewithtax
    methods(Static)
        function sol = hh_problem(par,sol)            
            beta = par.beta;
            agrid = par.agrid;
            alen = par.alen;
            zgrid = par.zgrid;
            zlen = par.zlen;
            pmat = par.pmat;
            r = par.r;
            w = par.w;
            tau = par.tau;
            ubi = par.ubi;
            phi = 0; % No borrowing

            v1 = nan(alen,zlen,3); % Value function for young, middle-aged, old
            a1 = nan(alen,zlen); % Savings (middle-aged to old)
            c1 = nan(alen,zlen); % Middle-aged consumption
            c2 = nan(alen,zlen); % Young consumption
            c3 = nan(alen,zlen); % Old consumption

            crit = 1e-6;
            maxiter = 10000;
            diff = 1;
            iter = 0;

            fprintf('------------Beginning Value Function Iteration.------------\n\n')

            % Initial guess for value function
            c0 = (1 + r) * agrid + w .* zgrid + ubi;
            v0 = modelwithtax.utility(c0, par) ./ (1 - beta);
            
            % Compute tax revenue and adjust tau for budget balance
            income = w * par.e(2) * zgrid; % Middle-aged labor income
            avg_income = mean(income); % Average labor income
            total_ubi = par.ubi * par.nage; % Total UBI payments (3 age groups)
            par.tau = total_ubi / avg_income; % Tax rate to balance budget
            tau = par.tau;
            fprintf('Adjusted tax rate for budget balance: tau = %.4f\n', tau)

            while diff > crit && iter < maxiter
                for i = 1:alen
                    if agrid(i) >= phi
                        for j = 1:zlen
                            % Middle-aged budget constraint
                            income = (1 - tau) * w * par.e(2) * zgrid(j); % After-tax labor income
                            asset_choices = agrid;
                            c = (1 + r) * agrid(i) + income + ubi - asset_choices;
        
                            ev = v0(:,:,3) * pmat(j,:)'; % Expected value (old age)
                            vall = modelwithtax.utility(c, par) + beta * ev;
                            vall(c <= 0) = -inf;
        
                            [vmax, ind] = max(vall);
        
                            v1(i,j,2) = vmax; % Middle-aged value
                            c1(i,j) = c(ind); % Middle-aged consumption
                            a1(i,j) = asset_choices(ind); % Savings to old age
        
                            % Young consumption (only UBI, no labor income)
                            c2(i,j) = ubi;
                            v1(i,j,1) = modelwithtax.utility(c2(i,j), par) + beta * mean(v0(i,:,2)); % Expected middle-aged value
        
                            % Old consumption (savings + UBI)
                            c3(i,j) = (1 + r) * a1(i,j) + ubi;
                            v1(i,j,3) = modelwithtax.utility(c3(i,j), par);
                        end
                    end
                end
        
                diff = norm(v1 - v0);
                v0 = v1;
                iter = iter + 1;
        
                if mod(iter, 25) == 0
                    fprintf('Iteration: %d.\n', iter)
                end
            end

            fprintf('\nConverged in %d iterations.\n\n', iter)
            fprintf('------------End of Value Function Iteration.------------\n')

            % Compute welfare (average utility of consumption)
            util_young = modelwithtax.utility(c2, par);
            util_middle = modelwithtax.utility(c1, par);
            util_old = modelwithtax.utility(c3, par);
            avg_utility = (mean(util_young(:)) + mean(util_middle(:)) + mean(util_old(:))) / par.nage;
            fprintf('Average utility of consumption: %.4f\n', avg_utility)

            sol.a = a1;
            sol.c = c1; % Middle-aged consumption
            sol.c2 = c2; % Young consumption
            sol.c3 = c3; % Old consumption
            sol.v = v1;
            sol.welfare = avg_utility; % Store welfare
            sol.tau = tau; % Store adjusted tax rate
        end

        function [par,sol] = firm_problem(par)
            delta = par.delta;
            alpha = par.alpha;
            r = par.r;
            k = ((r + delta) / alpha) ^ (1 / (alpha - 1));

            sol = struct();
            sol.k = k;
            par.w = (1 - alpha) * k ^ alpha;
        end
    end
end
