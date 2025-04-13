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

%%%% Since I have a Mac
%Assets = Assets(:,2:end)

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
Returns_GARCH_MV = zeros(121,1);

for i = 1 : size(Returns, 1)

    Returns_Help = Returns(1:i,:);
    %Estimating VARCOV period by period for each stock with garch
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

   
    objective = @(weights_MV) -(Mean*weights_MV - Rf(i)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize
    %objective = @(weights_MV) -(mean(Returns_Help_adj,1)*weights_MV - Rf(i)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize


    %Set constraints
    A = []; b = [];  % No inequality constraints in this case\
    Aeq = ones(1,No_Assets); beq = 1;  % weights sum to 1
    lb = zeros(No_Assets,1); ub = 0.5 * ones(No_Assets,1);  % 0 ≤ weights ≤ 0.5

    % Optimization
    options = optimset('Display', 'off');
    [w_MV, sr_MV] = fmincon(objective, weights_MV, A, b, Aeq, beq, lb, ub, [], options);
    w_MV(w_MV < 0.0001) = 0;        %set to 0 very small values
    w_MV = w_MV/sum(w_MV);        %Rebalance weights

    if i==1
        TC = sum(abs(w_MV))*0.0005;
    else 
        TC = sum(abs(w_MV-w_MV_TC))*0.0005;
    end
    
    w_MV_TC = w_MV;

    Returns_adj = Returns(i,:);
    Returns_adj(isnan(Returns_adj)) = 0;
    Ret_GARCH_MV = Returns_adj * w_MV - TC;
    Returns_GARCH_MV(i,1) = Ret_GARCH_MV;
    Wealth_GARCH_MV(i+1,1) = Wealth_GARCH_MV(i,1) * exp(Ret_GARCH_MV);


end

plot(Wealth_GARCH_MV);

%% Backtesting equally weighted portfolio
Wealth_ew(1,1) = [100];
Returns_ew = zeros(121,1);

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
    % Added to have returns as well as wealth
    Returns_ew(i,1) = Ret_ew;
    Wealth_ew(i+1,1) = Wealth_ew(i,1) * exp(Ret_ew);


end
plot(Wealth_ew);

%% Performance measures
% Create excess returns
Returns_ew_excess = Returns_ew - Rf
Returns_GARCH_MV_excess = Returns_GARCH_MV - Rf

% Average absolute monthly excess returns
Avg_monthly_ew = exp(mean(Returns_ew_excess))-1
Avg_monthly_GARCH_MV = exp(mean(Returns_GARCH_MV_excess))-1

% Average absolute annual excess returns
Avg_annual_ew = exp(mean(Returns_ew_excess)*12)-1
Avg_annual_GARCH_MV = exp(mean(Returns_GARCH_MV_excess)*12)-1

% Standard deviations
Std_ew = std(Returns_ew_excess) * sqrt(12)
Std_GARCH_MV = std(Returns_GARCH_MV_excess) * sqrt(12)

% Skewness
Skew_ew = skewness(Returns_ew_excess)
Skew_GARCH_MV = skewness(Returns_GARCH_MV_excess)

% Kurtosis
Kurt_ew = kurtosis(Returns_ew_excess)
Kurt_GARCH_MV = kurtosis(Returns_GARCH_MV_excess)

% Sharpe ratio
Sharpe_ew = (mean(Returns_ew_excess) / std(Returns_ew_excess)) * sqrt(12)
Sharpe_GARCH_MV = (mean(Returns_GARCH_MV_excess) / std(Returns_GARCH_MV_excess))* sqrt(12)

% Estimate betas
b_ew = regress(Returns_ew_excess, Market)
b_GARCH_MV = regress(Returns_GARCH_MV_excess, Market)

Treynor_ew = mean(Returns_ew_excess)*12/b_ew
Treynor_GARCH_MV = mean(Returns_GARCH_MV_excess)*12/b_GARCH_MV

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maximum Drawdown
Maximum_Drawdown_ew = 0
for i = 1 : size( Wealth_ew, 1 )
    High_Water_Mark = max( Wealth_ew( 1 : i,1 ))
    Drawdown = ( Wealth_ew( i, 1 ) - High_Water_Mark ) / High_Water_Mark
    Maximum_Drawdown_ew = min( Maximum_Drawdown_ew, Drawdown )
end

Maximum_Drawdown_GARCH_MV = 0
for i = 1 : size( Wealth_GARCH_MV, 1 )
    High_Water_Mark = max( Wealth_GARCH_MV( 1 : i,1 ))
    Drawdown = ( Wealth_GARCH_MV( i, 1 ) - High_Water_Mark ) / High_Water_Mark
    Maximum_Drawdown_GARCH_MV = min( Maximum_Drawdown_GARCH_MV, Drawdown )
end

%% Historical VAR
VAR_ew = 1 - exp(prctile( Returns_ew_excess, [ 1 ] ))
VAR_GARCH_MV = 1 - exp(prctile( Returns_GARCH_MV_excess, [ 1 ] ))

%% Shortfall measures
Shortfall_ew = 1 - exp(mean(Returns_ew_excess(Returns_ew_excess <= prctile(Returns_ew_excess, 1))))
Shortfall_GARCH_MV = 1 - exp(mean(Returns_GARCH_MV_excess(Returns_GARCH_MV_excess <= prctile(Returns_GARCH_MV_excess, 1))))


%% Add appropriate benchmark
index=readmatrix('C:\Users\emaho\Desktop\2 year lessons\AQAM\Assignment\index.csv'); 
%%%% Since I have a Mac
index = index(:,2:end)
index   = diff( log( index ) );

% Regressing for Alpha
reg1 = regress(Returns_ew, [ones(length(index), 1), index])
reg2 = regress(Returns_GARCH_MV, [ones(length(index), 1), index])

alpha_ew = exp(reg1(1)*12)-1
alpha_GARCH_MV = exp(reg2(1)*12)-1

% Tracking Error
TE_ew = std(Returns_ew - index) * sqrt(12)
TE_GARCH = std(Returns_GARCH_MV - index) * sqrt(12)

% Information Ratio
IR_ew = alpha_ew / TE_ew
IR_GARCH = alpha_GARCH_MV / TE_GARCH

%% Figure of Wealth
%Calculate Wealth investing in index
index_wealth = [100];
for i = 1 : size(index,1)
   index_wealth(i+1) = index_wealth(i)*exp(index(i));
end
% Create time vector from Feb 2015 to Feb 2025 with same length as data
startDate = datetime(2015, 2, 1);
endDate = datetime(2025, 2, 1);
time = linspace(startDate, endDate, length(Wealth_GARCH_MV))';

% Plot the three series
figure;
plot(time, Wealth_GARCH_MV, 'b-', 'LineWidth', 1.5); hold on;
plot(time, Wealth_ew, 'r-', 'LineWidth', 1.5);
plot(time, index_wealth, 'k-.', 'LineWidth', 1.5);
hold off;

% Labels and legend
xlabel('Time');
ylabel('Cumulative Performance');
title('Portfolio Performance Comparison');
legend('Mean Variance with Garch', 'Equally Weighted', ...
       'STOXX Europe Total Market Aerospace & Defense', ...
       'Location', 'northwest');

grid on;
xlim([startDate endDate]);

saveas(gcf, 'portfolio_comparison.png');


%% Factor Loadings GARCH MV

Fund_Exposure_Results = zeros( size( Returns_GARCH_MV_excess, 1 ) - 47, 5 );   %take only 4 years as we estimate only 4 parameters (3 Factors + intercept)

for i = 48 : size(Returns_GARCH_MV_excess)
    Returns_Help = Returns_GARCH_MV_excess((i-47):i);
    Factors_Help = Factors((i-47):i,1:3);
    [b,bint,r,rint,stats] = regress( Returns_Help, [ ones( size( Returns_Help, 1 ), 1 ) Factors_Help ] );
    Factors_Exposure_Results_GARCH_MV( i - 47, : ) = b';
end

Factors_Exposure_Results_GARCH_MV(:,1) = Factors_Exposure_Results_GARCH_MV(:,1)*12;    %annualize alpha for better graphic rapresentation


% Illustrate style drift using a graph
plot( Factors_Exposure_Results_GARCH_MV );
legend( 'Alpha (annualized)', 'Market', 'SMB', 'HML' );
xlabel( 'Time' );
ylabel( 'Parameter Estimate' );
title('Factor loadings through time of Mean Variance Portfolio');

saveas(gcf, 'factor_loadings_MV.png');

%% Factor Loadings Equally weighted

Fund_Exposure_Results = zeros( size( Returns_ew_excess, 1 ) - 47, 5 );   %take only 4 years as we estimate only 4 parameters (3 Factors + intercept)

for i = 48 : size(Returns_ew_excess)
    Returns_Help = Returns_ew_excess((i-47):i);
    Factors_Help = Factors((i-47):i,1:3);
    [b,bint,r,rint,stats] = regress( Returns_Help, [ ones( size( Returns_Help, 1 ), 1 ) Factors_Help ] );
    Factors_Exposure_Results_ew( i - 47, : ) = b';
end

Factors_Exposure_Results_ew(:,1) = Factors_Exposure_Results_ew(:,1)*12;

% Illustrate style drift using a graph
plot( Factors_Exposure_Results_ew );
legend( 'Alpha (annualized)', 'Market', 'SMB', 'HML' );
xlabel( 'Time' );
ylabel( 'Parameter Estimate' );
title('Factor loadings through time of Equally Weighted Portfolio');


saveas(gcf, 'factor_loadings_ew.png');







