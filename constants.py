from datetime import datetime

TIMESTR = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
GRID_ROOT = 'gridsearch_results'
dataset_names = ['compas', 'law-gender', 'adult', 'law-race', 'wiki']

name_map = {
    'noise_DirectRankerAdv': r'ADV DR$^\star$',
    'noise_DirectRankerAdvFlip': r'ADV FFDR$^\star$',
    'noise_DirectRankerSymFlip': r'FFDR$^\star$',
    'noise_FairListNet': r'DELTR$^\star$',
    'noise_DebiasClassifier': r'Clas.$^\star$',
    'nonoise_DirectRankerAdv': 'ADV DR',
    'nonoise_DirectRankerAdvFlip': 'ADV FFDR',
    'nonoise_DirectRankerSymFlip': 'FFDR',
    'nonoise_FairListNet': 'DELTR',
    'nonoise_DebiasClassifier': 'Clas.',
    'baseline': 'Base.'
}

name_csv = {
    'noise_DirectRankerAdv': 'ADV DR $^\star$',
    'noise_DirectRankerAdvFlip': 'ADV FFDR $^\star$',
    'noise_DirectRankerSymFlip': 'FFDR $^\star$',
    'noise_FairListNet': 'DELTR $^\star$',
    'noise_DebiasClassifier': 'Clas. $^\star$',
    'nonoise_DirectRankerAdv': 'ADV DR',
    'nonoise_DirectRankerAdvFlip': 'ADV FFDR',
    'nonoise_DirectRankerSymFlip': 'FFDR',
    'nonoise_FairListNet': 'DELTR',
    'nonoise_DebiasClassifier': 'Clas.',
    'baseline': 'Base.'
}

name_2d = {
    'noise_DirectRankerAdv': 'ADV DR n.',
    'noise_DirectRankerAdvFlip': 'ADV FFDR n.',
    'noise_DirectRankerSymFlip': 'FFDR n.',
    'noise_FairListNet': 'DELTR n.',
    'noise_DebiasClassifier': 'Clas. n.',
    'nonoise_DirectRankerAdv': 'ADV DR',
    'nonoise_DirectRankerAdvFlip': 'ADV FFDR',
    'nonoise_DirectRankerSymFlip': 'FFDR',
    'nonoise_FairListNet': 'DELTR',
    'nonoise_DebiasClassifier': 'Clas.',
    'baseline': 'Base.'
}

colors = [(114 / 255, 147 / 255, 203 / 255),    # blue
          (225 / 255, 151 / 255, 76 / 255),     # orange
          (132 / 255, 186 / 255, 91 / 255),     # green
          (211 / 255, 94 / 255, 96 / 255),      # red
          (128 / 255, 133 / 255, 133 / 255),    # black
          (144 / 255, 103 / 255, 167 / 255),    # purple
          (171 / 255, 104 / 255, 87 / 255),     # wine
          (204 / 255, 194 / 255, 16 / 255)      # gold
          ]

colors_points = [(57 / 255, 106 / 255, 177 / 255),    # blue
          (218 / 255, 124 / 255, 48 / 255),     # orange
          (132 / 255, 186 / 255, 91 / 255),     # green
          (204 / 255, 37 / 255, 41 / 255),      # red
          (83 / 255, 81 / 255, 84 / 255),    # black
          (107 / 255, 76 / 255, 154 / 255),    # purple
          (171 / 255, 104 / 255, 87 / 255),     # wine
          (204 / 255, 194 / 255, 16 / 255)      # gold
          ]