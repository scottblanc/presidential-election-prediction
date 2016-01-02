import numpy as np
import pandas as pd
from datetime import time

# calculate spread in terms of how many more votes democrats received over republicans
def add_vote_spread(df,col_name):
  df[col_name] = df['Dem'] - df['Rep']


# add vote spread and use last poll in each month
def enrich_poll_preds(poll_preds):
  add_vote_spread(poll_preds,'Spread')
  poll_preds['Month'] = map(lambda x: int(x.strftime('%m')),poll_preds["Date"])
  poll_preds = poll_preds.sort(['State','Pollster','Date']).groupby(['State','Month','Pollster'], sort=False).last().reset_index()
  return poll_preds


# calculate errors of each pollster by state
def calculate_pollster_errors(state_polls,state_results):
  pollster_avg = pd.merge(state_polls, state_results, how='left', on=['State'],suffixes=['_p', '_a'])
  pollster_avg['PollsterError'] = (pollster_avg['Spread'] - pollster_avg['ActualSpread']).abs()
  pollster_avg = pollster_avg.groupby(['Pollster']).mean().reset_index().drop(['Month'], 1)
  return pollster_avg[['Pollster','PollsterError']]


# calculate aggregate polling data using weighted averages across individual polls
def calc_pollster_weighted_spreads(state_polls,by_month=False):
  state_polls_averages = state_polls.copy()
  state_polls_averages["Pollster_Weight"] = state_polls_averages["PollsterError"] ** -2
  state_polls_averages["PredSpread"] = state_polls_averages["Pollster_Weight"] * state_polls_averages["Spread"]

  if by_month:
    state_polls_averages = state_polls_averages.groupby(["State","Month"]).sum().reset_index()
  else:
    state_polls_averages = state_polls_averages.groupby(["State"]).sum().reset_index()
  state_polls_averages['PredSpread'] = state_polls_averages['PredSpread'] / state_polls_averages['Pollster_Weight']

  if by_month:
    state_polls_averages = state_polls_averages[['State','Month','PredSpread']]
  else:
    state_polls_averages = state_polls_averages[['State','PredSpread']]
  return state_polls_averages


# use state predictions and electoral college data to predict electoral outcomes for each state
def pred_state_electoral_outcomes(state_preds,ec):
  pred = pd.merge(state_preds,ec , how='inner', on=['State'])
  pred['Dem'] = np.where(pred['PredSpread']>0, pred['Electors'], 0)
  pred['Rep'] = np.where(pred['PredSpread']<0, pred['Electors'], 0)

  #split electors if there is a tie
  pred['Dem'] = np.where(pred['PredSpread']==0, pred['Electors']/2, pred['Dem'])
  pred['Rep'] = np.where(pred['PredSpread']==0, pred['Electors']/2, pred['Rep'])
  return pred


# use most recent polling data for each pollster to predict spreads
def predict_state_spreads_using_most_recent_polling_data(state_polls,pollster_errors):
  state_polls = pd.merge(state_polls, pollster_errors, how='left', on=['Pollster'])
  state_polls["PollsterError"].fillna(4, inplace=True)

  most_recent_state_polls = state_polls.sort(['State','Pollster','Date']).groupby(['State','Pollster'],sort=False).last().reset_index()
  most_recent_state_polls_averages = calc_pollster_weighted_spreads(most_recent_state_polls)
  #most_recent_state_polls_averages = most_recent_state_polls_averages.sort(['State','Month']).groupby(['State'],sort=False).last().reset_index()
  state_preds = most_recent_state_polls_averages[['State','PredSpread']]
  return state_preds


if __name__ == "__main__":

  #load actual 2008 results by state
  state_results_2008 = pd.read_csv("data/2008-results.csv")
  add_vote_spread(state_results_2008,'ActualSpread')

  #load actual 2012 results by state
  state_results_2012 = pd.read_csv("data/2012-results.csv")
  add_vote_spread(state_results_2012,'ActualSpread')

  #load 2008 polling data by state
  raw_state_polls_2008 = pd.read_csv("data/2008-polls.csv",parse_dates=["Date"])

  #enrich polling data
  state_polls_2008 = enrich_poll_preds(raw_state_polls_2008)

  #load 2012 polling data by state
  raw_state_polls_2012 = pd.read_csv("data/2012-polls.csv",parse_dates=["Date"])

  #enrich polling data
  state_polls_2012 = enrich_poll_preds(raw_state_polls_2012)

  #extract only most recent 2008 polling data 
  most_recent_state_polls_2008 = state_polls_2008.sort(['State','Pollster','Date']).groupby(['State','Pollster'],sort=False).last().reset_index()

  #calculate 2008 polling errors 
  pollster_errors_2008 = calculate_pollster_errors(most_recent_state_polls_2008,state_results_2008)

  #predict spreads by state for 2012
  state_preds_2012 = predict_state_spreads_using_most_recent_polling_data(state_polls_2012,pollster_errors_2008)

  #load 2012 electoral college data
  ec_2012 = pd.read_csv("data/2012-electoral-college.csv")

  #predict 2012 electoral outcomes
  state_pred_outcomes_2012 = pred_state_electoral_outcomes(state_preds_2012,ec_2012)

  #merge predicted and actual
  state_pred_vs_actual_2012 = pd.merge(state_pred_outcomes_2012, state_results_2012, how='inner', on=['State'],suffixes=['_p', '_a'])
  state_pred_vs_actual_2012['PredictionError'] = state_pred_vs_actual_2012['ActualSpread'] - state_pred_vs_actual_2012['PredSpread']
  state_pred_vs_actual_2012['Pred'] = state_pred_vs_actual_2012['PredSpread'] / state_pred_vs_actual_2012['PredSpread'].abs()
  state_pred_vs_actual_2012['Actual'] = state_pred_vs_actual_2012['ActualSpread'] / state_pred_vs_actual_2012['ActualSpread'].abs()
  state_pred_vs_actual_2012['Correct'] = state_pred_vs_actual_2012['Actual'] == state_pred_vs_actual_2012['Pred']

  #write results to file
  state_pred_vs_actual_2012.to_csv('2012_election_state_predicted_vs_actual', sep='\t')

  mean_pred_error = state_pred_vs_actual_2012['PredictionError'].abs().mean()
  num_correct_states = state_pred_vs_actual_2012['Correct'].sum()
   
  print "[Model results] Mean absolute error: %.2f, State voting accuracy: %.2f" % (mean_pred_error,num_correct_states*100.0/len(state_pred_vs_actual_2012))

