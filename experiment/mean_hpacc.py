import pandas as pd

path = '../'


def get_samples_from_disk():
    df = pd.read_csv(path+"data/hp_config.csv", dtype={'scope': float, 'width': float, 'damp': float})
    configurations = df.values
    return configurations


dfs = [pd.read_csv(path+'results/hpopt_01.csv'),
       pd.read_csv(path+'results/hpopt_02.csv'),
       pd.read_csv(path+'results/hpopt_03.csv')
      ]

meanacc = pd.DataFrame(columns=["val_acc", "test_acc", "strategy", "scope", "width", "damp"])

configs = get_samples_from_disk()
for scope, width, damp in configs:
    for strategy in ['converged', 'toeplitz']:
        val_accs = []
        test_accs = []
        for df in dfs:
            acc = df.loc[(df['scope'] == scope) & (df['width'] == width) &
                         (df['damp'] == damp) & (df['strategy'] == strategy)]
            val_accs.append(acc['val_acc'].values[0])
            test_accs.append(acc['test_acc'].values[0])
        val_mean = sum(val_accs) / len(val_accs)
        test_mean = sum(test_accs) / len(test_accs)
        #print(scope, width, damp)
        #print(val_accs, val_mean, test_accs, test_mean)
        meanacc = meanacc.append(
            {'val_acc': round(val_mean, 2), 'test_acc': round(test_mean, 2), 'strategy': strategy, 'scope': int(scope), 'width': int(width),
             'damp': damp}, ignore_index=True)

        meanacc = meanacc.sort_values(by='val_acc', ascending=False)
        meanacc.to_csv(path_or_buf=f"{path}results/mean_acc.csv", index=False)


