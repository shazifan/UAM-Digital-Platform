import pandas_alive
import pandas as pd
covid_df = pd.read_csv('covid19.csv', index_col=0, parse_dates=[0])
covid_df.sum(axis=1).fillna(0).plot_animated(filename='examples/example-bar-chart.gif',kind='bar',
                                             period_label={'x':0.1,'y':0.9},
                                             enable_progress_bar=True, steps_per_period=2, interpolate_period=True, period_length=200)
