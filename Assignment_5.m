% MBF: Applied Quantitative Asset Management
% Spring 2025
%
% AQAM: Course Assignment
% 
%European Defense Sector Equity Fund
%-------------------------------------------------------------------------

clear; clc;
Assets = xlsread( 'C:\Users\emaho\Desktop\2 year lessons\AQAM\Assignment\assignment_prices.xlsx' );
Factors=csvread('C:\Users\emaho\Desktop\2 year lessons\AQAM\Assignment\Europe_3_FF_Factors.csv',1,1); 
Factors = log(1+Factors/100);
Factors = Factors(2:end,:);
Market=Factors( :, 1 );                         % Defines the market excess return
SMB=Factors( :, 2 );                            % Defines the Small-Minus-Big factor
HML=Factors( :, 3 );                            % Defines the High-Minus-Low factor
Rf = Factors(:,4);
No_Assets = size( Assets, 2 )-1;
Returns   = diff( log( Assets ) );
Returns(:, 12) = [];  % Remove 12th stock as it has too few observations


% Calculate Standard Statistics            %these 
Mean = mean(Returns,'omitnan');
Median = median(Returns);
Volatility = std(Returns,'omitnan');
Returns_for_corr = Returns;
Returns_for_corr(any(isnan(Returns_for_corr), 2), :) = [];
Corr = corr(Returns_for_corr);
Cov = cov(Returns_for_corr);

Skew = skewness(Returns);
Kurt = kurtosis(Returns);

Forecast_Variance = zeros(size(Returns, 2),1);

%Estimating Garch Model to use for VARCOV
Models_GARCH = cell(size(Returns, 2), 1);  % Pre-allocate cell array to store models

for i = 1:size(Returns,2)

    Returns_Help = Returns(:,i) - mean( Returns(:,i),'omitnan'); %here we just demean the returns, CER
    mdl = garch(1,1);
    mdl = estimate(mdl, Returns_Help);

    Models_GARCH{i} = mdl;                  % Store model in cell array



end


%Estimation of VARCOV in final period
for i = 1:size(Returns,2)
    Returns_Help = Returns(:,i) - mean( Returns(:,i),'omitnan');
    Returns_Help = Returns_Help(~isnan(Returns_Help));  % Remove NaN values
    Fore = forecast(Models_GARCH{i},1,'Y0',Returns_Help);
    Forecast_Variance(i,1) = Fore;
    
end
Forecast_Volatility = sqrt(Forecast_Variance);

VARCOV = Forecast_Volatility * Forecast_Volatility' .* Corr;

%Mean-Variance optimization
weights_MV = ones(No_Assets,1)/No_Assets;

% Objective function (Sharpe Ratio)
objective = @(weights_MV) -(Mean*weights_MV - Rf(end)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize

%Set constraints
A = []; b = [];  % No inequality constraints in this case\
Aeq = ones(1,No_Assets); beq = 1;  % weights sum to 1
lb = zeros(No_Assets,1); ub = 0.5 * ones(No_Assets,1);  % 0 ≤ weights ≤ 0.5

% Optimization
options = optimset('Display', 'off');
[w_MV, sr_MV] = fmincon(objective, weights_MV, A, b, Aeq, beq, lb, ub, [], options);

Er_MV = Mean * weights_MV;
Vol_MV = sqrt(weights_MV' * VARCOV * weights_MV);
SR_MV = -sr_MV * sqrt(12);   %annualize the SR

%% Backtesting GARCH VARCOV MV
Wealth_GARCH_MV(1,1) = [100];
for i = 1 : size(Returns, 1)

    Returns_Help = Returns(1:i,:);
    %Estimating VARCOV period by period with garch
    for j = 1 : No_Assets

        %Returns_demeaned = Returns_Help(:,j) - mean(Returns_Help(:,j));
        Returns_demeaned = Returns_Help(:,j) - Mean(:,j);

        Returns_demeaned = Returns_demeaned(~isnan(Returns_demeaned));  % Remove NaN values
        if isempty(Returns_demeaned)
            Forecast_Variance(j,1) = 999999999999;   %make variance very high if NaN so we don't invest in it
   
        else 
            Fore = forecast(Models_GARCH{j},1,'Y0',Returns_demeaned);
            Forecast_Variance(j,1) = Fore;
        end

    
    end
    
    Forecast_Volatility = sqrt(Forecast_Variance);

    VARCOV = Forecast_Volatility * Forecast_Volatility' .* Corr;

    weights_MV = ones(No_Assets,1)/No_Assets;

    Returns_Help_adj = Returns_Help;
    Returns_Help_adj(isnan(Returns_Help_adj)) = 0;    
    objective = @(weights_MV) -(Mean*weights_MV - Rf(i)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize
    %objective = @(weights_MV) -(mean(Returns_Help_adj,1)*weights_MV - Rf(i)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize


    %Set constraints
    A = []; b = [];  % No inequality constraints in this case\
    Aeq = ones(1,No_Assets); beq = 1;  % weights sum to 1
    lb = zeros(No_Assets,1); ub = 0.3 * ones(No_Assets,1);  % 0 ≤ weights ≤ 0.5

    % Optimization
    options = optimset('Display', 'off');
    [w_MV, sr_MV] = fmincon(objective, weights_MV, A, b, Aeq, beq, lb, ub, [], options);
    w_MV(w_MV < 0.001) = 0;        %set to 0 very small values

    if i==1
        TC = sum(abs(w_MV))*0.0005;
    else 
        TC = sum(abs(w_MV-w_MV_TC))*0.0005;
    end
    
    w_MV_TC = w_MV;

    Returns_adj = Returns(i,:);
    Returns_adj(isnan(Returns_adj)) = 0;
    Ret_GARCH_MV = Returns_adj * w_MV - TC;
    Wealth_GARCH_MV(i+1,1) = Wealth_GARCH_MV(i,1) * exp(Ret_GARCH_MV);


end

plot(Wealth_GARCH_MV);

%% Backtesting equally weighted portfolio
Wealth_ew(1,1) = [100];

for i = 1:size(Returns,1)
    cols_with_nan = isnan(Returns(i, :));
    weights_ew = ones(size(Returns,2),1)/size(Returns,2);
    weights_ew(cols_with_nan) = 0;
    weights_ew = weights_ew/sum(weights_ew);
  

    if i==1
        TC = sum(abs(weights_ew))*0.0005;
    else 
        TC = sum(abs(weights_ew-weights_ew_TC))*0.0005;
    end
    
    weights_ew_TC = weights_ew;
    Returns_cleaned = Returns(i,:);
    Returns_cleaned(isnan(Returns_cleaned)) = 0;
    Ret_ew = Returns_cleaned * weights_ew - TC;
    Wealth_ew(i+1,1) = Wealth_ew(i,1) * exp(Ret_ew);


end
plot(Wealth_ew);

%%
% %% Backtesting Historical VARCOV MV
% 
% Wealth_Hist_MV(1,1) = [100];
% 
% for i = 13 : size(Returns, 1)      %at least 1 year to calculate varcov
% 
%     Returns_Help = Returns(1:i,:)
%     weights_MV = ones(No_Assets,1)/No_Assets;
%     Varcov_hist = cov(Returns_Help);
%     Varcov_hist(isnan(Varcov_hist)) = 9999999999999;   %setting 
%     objective = @(weights_MV) -(Mean*weights_MV - Rf(i)) / sqrt(weights_MV' * Cov * weights_MV);  % Negative Sharpe Ratio to minimize
% 
%     %Set constraints
%     A = []; b = [];  % No inequality constraints in this case\
%     Aeq = ones(1,No_Assets); beq = 1;  % weights sum to 1
%     lb = zeros(No_Assets,1); ub = 0.5 * ones(No_Assets,1);  % 0 ≤ weights ≤ 0.5
% 
%     % Optimization
%     options = optimset('Display', 'off');
%     [w_MV_h, sr_MV] = fmincon(objective, weights_MV, A, b, Aeq, beq, lb, ub, [], options);
%     w_MV_h(w_MV_h < 0.001) = 0;        %set to 0 very small values
% 
% 
% end
% 



