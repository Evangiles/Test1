Overview
Your task is to predict the stock market returns as represented by the excess returns of the S&P 500 while also managing volatility constraints. Your work will test the Efficient Market Hypothesis and challenge common tenets of personal finance.

Start

2 months ago
Close
a month to go
Merger & Entry
Description
Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?

This competition is about more than predictive modeling. Predicting market returns challenges the assumptions of market efficiency. Your work could help reshape how investors and academics understand financial markets. Participants could uncover signals others overlook, develop innovative strategies, and contribute to a deeper understanding of market behavior—potentially rewriting a fundamental principle of modern finance. Most investors don’t beat the S&P 500. That failure has been used for decades to prop up EMH: If even the professionals can’t win, it must be impossible. This observation has long been cited as evidence for the Efficient Market Hypothesis the idea that prices already reflect all available information and no persistent edge is possible. This story is tidy, but reality is less so. Markets are noisy, messy, and full of behavioral quirks that don’t vanish just because academic orthodoxy said they should.

Data science has changed the game. With enough features, machine learning, and creativity, it’s possible to uncover repeatable edges that theory says shouldn’t exist. The real challenge isn’t whether they exist—it’s whether you can find them and combine them in a way that is robust enough to overcome frictions and implementation issues.

Our current approach blends a handful of quantitative models to adjust market exposure at the close of each trading day. It points in the right direction, but with a blurry compass. Our model is clearly a sub-optimal way to model a complex, non-linear, adaptive system. This competition asks you to do better: to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint. We’ll provide daily data that combines public market information with our proprietary dataset, giving you the raw material to uncover patterns most miss.

Unlike many Kaggle challenges, this isn’t just a theoretical exercise. The models you build here could be valuable in live investment strategies. And if you succeed, you’ll be doing more than improving a prediction engine—you’ll be helping to demonstrate that financial markets are not fully efficient, challenging one of the cornerstones of modern finance, and paving the way for better, more accessible tools for investors.

Evaluation
The competition's metric is a variant of the Sharpe ratio that penalizes strategies that take on significantly more volatility than the underlying market or fail to outperform the market's return. The metric code is available here.

Submission File
You must submit to this competition using the provided evaluation API, which ensures that models do not peek forward in time. For each trading day, you must predict an optimal allocation of funds to holding the S&P500. As some leverage is allowed, the valid range covers 0 to 2. See this example notebook for more details.

Timeline
This is a forecasting competition with an active training phase and a separate forecasting phase where models will be run against real market returns.
Training Timeline:
September 16, 2025 - Start Date.
December 8, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
December 8, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
December 15, 2025 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Forecasting Timeline:
Starting after the final submission deadline there will be periodic updates to the leaderboard to reflect market data updates that will be run against selected notebooks.

June 16, 2026 - Competition End Date
Prizes
1st Place - $50,000
2nd Place - $25,000
3rd Place - $10,000
4th Place - $5,000
5th Place - $5,000
6th Place - $5,000
Code Requirements


Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 8 hours run-time
GPU Notebook <= 8 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Forecasting Phase
The run-time limits for both CPU and GPU notebooks will be extended to 9 hours during the forecasting phase. You must ensure your submission completes within that time.

The extra hour is to help protect against time-out failures due to the extended size of the test set. You are still responsible for ensuring your submission completes within the 9 hour limit, however. See the Data page for details on the extended test set during the forecasting phase.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Citation
Blair Hull, Petra Bakosova, Laurent Lanteigne, Aishvi Shah, Euan C Sinclair, Petri Fast, Will Raj, Harold Janecek, Sohier Dane, and Addison Howard. Hull Tactical - Market Prediction. https://kaggle.com/competitions/hull-tactical-market-prediction, 2025. Kaggle.


Dataset Description
This competition challenges you to predict the daily returns of the S&P 500 index using a tailored set of market data features.

Competition Phases and Data Updates
The competition will proceed in two phases:

A model training phase with a test set of six months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
A forecasting phase with a test set to be collected after submissions close. You should expect the scored portion of the test set to be about the same size as the scored portion of the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set. This includes trading days before the submission deadline, which will not be scored. The first date_id served by the API will remain constant throughout the competition.

Files
train.csv Historic market data. The coverage stretches back decades; expect to see extensive missing values early on.

date_id - An identifier for a single trading day.
M* - Market Dynamics/Technical features.
E* - Macro Economic features.
I* - Interest Rate features.
P* - Price/Valuation features.
V* - Volatility features.
S* - Sentiment features.
MOM* - Momentum features.
D* - Dummy/Binary features.
forward_returns - The returns from buying the S&P 500 and selling it a day later. Train set only.
risk_free_rate - The federal funds rate. Train set only.
market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. Train set only.
test.csv A mock test set representing the structure of the unseen test set. The test set used for the public leaderboard set is a copy of the last 180 date IDs in the train set. As a result, the public leaderboard scores are not meaningful. The unseen copy of this file served by the evaluation API may be updated during the model training phase.

date_id
[feature_name] - The feature columns are the same as in train.csv.
is_scored - Whether this row is included in the evaluation metric calculation. During the model training phase this will be true for the first 180 rows only. Test set only.
lagged_forward_returns - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
lagged_risk_free_rate - The federal funds rate, provided with a lag of one day.
lagged_market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.
kaggle_evaluation/ Files used by the evaluation API. See the demo submission for an illustration of how to use the API.

Once the competition ends, we will periodically publish our data on our website, and you're welcome to use it for your own trading

Files
13 files

Size
12.44 MB

Type
py, csv, proto

License
Subject to Competition Rules

test.csv(16.75 kB)

10 of 99 columns


date_id

D1

D2

D3

D4

D5

D6

D7

D8

D9
10

total values
10

total values
10

total values
10

total values
10

total values
10

total values
10

total values
10

total values
10

total values
10

total values
8980
0
0
0
0
1
0
0
1
0
8981
0
0
0
0
1
0
0
1
0
8982
0
0
0
0
1
0
0
0
1
8983
0
0
0
0
1
0
0
0
1
8984
0
0
0
0
0
0
1
0
1
8985
0
0
0
0
0
0
0
0
0
8986
0
0
0
0
0
0
0
0
0
8987
0
0
1
0
0
0
0
0
0
8988
0
0
0
0
0
0
0
0
0
8989
0
0
0
0
0
0
0
0
0
No more data to show
Data Explorer
12.44 MB

kaggle_evaluation

test.csv

train.csv

Summary
13 files

197 columns


Download All
kaggle competitions download -c hull-tactical-market-prediction
Download data

Metadata
License
Subject to Competition Rules