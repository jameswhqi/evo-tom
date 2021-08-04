#%%
import json
import os

from funcy import autocurry, lmap
import numpy as np

#%%
def load_data(file_name):
    with open('data/' + file_name, 'r') as fh:
        return json.load(fh)

def ntrial(subject):
    return len(subject['trials'])

@autocurry
def correct_rate(correct_label, response_label, subject):
    return np.mean([t[correct_label] == t[response_label] for t in subject['trials']])

correct_pred_rate = correct_rate('correctPrediction', 'prediction')
correct_choice_rate = correct_rate('correctChoice', 'playerChoice')

#%%
file_names = (f for f in os.listdir('data') if f.startswith('sona'))
subjects = lmap(load_data, file_names)

ntrials = lmap(ntrial, subjects)
correct_pred_rates = lmap(correct_pred_rate, subjects)
correct_choice_rates = lmap(correct_choice_rate, subjects)

#%%
print('Correct prediction rates:')
print('  Mean:', np.mean(correct_pred_rates))
print('  SE:', np.std(correct_pred_rates, ddof=1) / np.sqrt(len(correct_pred_rates)))
print()
print('Correct choice rates:')
print('  Mean:', np.mean(correct_choice_rates))
print('  SE:', np.std(correct_choice_rates, ddof=1) / np.sqrt(len(correct_choice_rates)))
