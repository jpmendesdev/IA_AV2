import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NeuralNetWorks import Perceptron, ADALINE, MultilayerPerceptron

def labels_to_pm1(y):
    y = y.copy().reshape(-1)
    unique = np.unique(y)
    if set(unique) == {-1, 1}: return y
    if set(unique) <= {0, 1}: return np.where(y == 0, -1, 1)
    if set(unique) <= {1, 2}: return np.where(y == 1, -1, 1)
    lo = np.sort(unique)[0]
    return np.where(y == lo, -1, 1)

def confusion_matrix_manual(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TP, FN],[FP, TN]]), {'TP':TP,'TN':TN,'FP':FP,'FN':FN}

def metrics_from_counts(counts):
    TP, TN, FP, FN = counts['TP'], counts['TN'], counts['FP'], counts['FN']
    total = TP+TN+FP+FN
    acc = (TP+TN)/total if total>0 else 0.0
    sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    f1 = (2*prec*sens)/(prec+sens) if (prec+sens)>0 else 0.0
    return {'accuracy':acc,'sensitivity':sens,'specificity':spec,'precision':prec,'f1':f1}

def train_test_split_random(X, y, test_ratio=0.2, rng=None):
    if rng is None: rng = np.random.RandomState()
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(N*test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def run_experiments(csv_path="spiral_d.csv", R=50, seed=0,
                    mlp_topologies_under_over = [[2],[8],[40]]):
    hp = {
        'Perceptron': {'lr':1e-3,'max_epochs':200},
        'ADALINE': {'lr':1e-3,'max_epoch':200,'tol':1e-5},
        'MLP': {'lr':5e-3,'max_epoch':300,'tol':1e-6}
    }
    data = np.loadtxt(csv_path, delimiter=',')
    X_raw = data[:, :-1]
    y_raw = data[:, -1]
    y_pm1 = labels_to_pm1(y_raw)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-12
    X_norm = (X_raw - X_mean)/X_std
    plt.figure(figsize=(6,5))
    plt.scatter(X_norm[y_pm1==1,0], X_norm[y_pm1==1,1], marker='s', s=40, label='+1')
    plt.scatter(X_norm[y_pm1==-1,0], X_norm[y_pm1==-1,1], marker='o', s=40, label='-1')
    plt.title('Dispersão inicial (spiral)')
    plt.xlabel('x1'); plt.ylabel('x2'); plt.legend(); plt.tight_layout(); plt.show()
    metrics_names = ['accuracy','sensitivity','specificity','precision','f1']
    models = ['Perceptron','ADALINE','MLP']
    results = {m:{mm:[] for mm in metrics_names} for m in models}
    confusions = {m:[] for m in models}
    rng_global = np.random.RandomState(seed)
    mlp_learning_histories = {}
    for model_name in models:
        print(f"\n=== Treinando modelo: {model_name} ===")
        last_w = None; last_Xtr=None; last_ytr=None
        for r in range(R):
            rng = np.random.RandomState(rng_global.randint(0,2**31-1))
            Xtr, ytr, Xte, yte = train_test_split_random(X_norm, y_pm1, test_ratio=0.2, rng=rng)
            Xtr_T = Xtr.T; Xte_T = Xte.T
            ytr_vec = ytr.reshape(-1); yte_vec = yte.reshape(-1)
            if model_name == 'Perceptron':
                perc = Perceptron(Xtr_T, ytr_vec, learning_rate=hp['Perceptron']['lr'], plot=False)
                res = perc.fit(max_epochs=hp['Perceptron']['max_epochs'])
                preds = perc.predict(Xte_T)
                if r==0: perc_hist = res['errors_history']
                last_w = perc.w; last_Xtr = Xtr; last_ytr = ytr_vec
            elif model_name == 'ADALINE':
                ad = ADALINE(Xtr_T, ytr_vec, learning_rate=hp['ADALINE']['lr'],
                             max_epoch=hp['ADALINE']['max_epoch'], tol=hp['ADALINE']['tol'], plot=False)
                res = ad.fit()
                preds = ad.predict(Xte_T)
                if r==0: ad_hist = res['eqm_history']
                last_w = ad.w; last_Xtr = Xtr; last_ytr = ytr_vec
            elif model_name == 'MLP':
                Ytr = ytr_vec.reshape(1,-1)
                mlp = MultilayerPerceptron(Xtr_T, Ytr, topology=[8],
                                           learning_rate=hp['MLP']['lr'],
                                           tol=hp['MLP']['tol'],
                                           max_epoch=hp['MLP']['max_epoch'])
                res = mlp.fit()
                out = mlp.predict(Xte_T)
                preds = np.where(out.flatten() >= 0, 1, -1)
                if r==0: mlp_hist = res['eqm_history']

            cm, counts = confusion_matrix_manual(yte_vec, preds)
            confusions[model_name].append(cm)
            met = metrics_from_counts(counts)
            for mm in metrics_names:
                results[model_name][mm].append(met[mm])
            if (r+1) % max(1, R//5) == 0:
                print(f"  rodada {r+1}/{R} — acc={met['accuracy']:.4f}")
        if model_name in ['Perceptron','ADALINE'] and last_w is not None:
            w0,w1,w2 = last_w.flatten()
            plt.figure(figsize=(6,5))
            plt.scatter(last_Xtr[last_ytr==1,0], last_Xtr[last_ytr==1,1], marker='s', s=40, label='+1')
            plt.scatter(last_Xtr[last_ytr==-1,0], last_Xtr[last_ytr==-1,1], marker='o', s=40, label='-1')
            x_vals = np.linspace(np.min(last_Xtr[:,0]), np.max(last_Xtr[:,0]), 200)
            y_vals = -(w1/w2)*x_vals - (w0/w2)
            plt.plot(x_vals, y_vals, 'g--', label='reta separação (última rodada)')
            plt.title(f'{model_name} - reta encontrada (última rodada)')
            plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
        if model_name == 'Perceptron' and 'perc_hist' in locals():
            plt.figure(); plt.plot(perc_hist); plt.title('Perceptron - Erros por época'); plt.xlabel('época'); plt.ylabel('n erros'); plt.grid(); plt.show()
        if model_name == 'ADALINE' and 'ad_hist' in locals():
            plt.figure(); plt.plot(ad_hist); plt.title('ADALINE - EQM por época'); plt.xlabel('época'); plt.ylabel('EQM'); plt.grid(); plt.show()
        if model_name == 'MLP' and 'mlp_hist' in locals():
            plt.figure(); plt.plot(mlp_hist); plt.title('MLP - EQM por época'); plt.xlabel('época'); plt.ylabel('EQM'); plt.grid(); plt.show()
        cm_mean = np.sum(confusions[model_name], axis=0)
        plt.figure(figsize=(5,4)); sns.heatmap(cm_mean, annot=True, fmt='d', cmap='Blues'); plt.title(f'Confusão acumulada — {model_name}'); plt.tight_layout(); plt.show()

        data_to_plot = [results[model_name][m] for m in metrics_names]
        plt.figure(figsize=(7,5)); plt.boxplot(data_to_plot, tick_labels=metrics_names); plt.title(f'Boxplot métricas — {model_name}'); plt.grid(); plt.tight_layout(); plt.show()
    print("\n=== Exploração MLP: underfitting e overfitting ===")
    for topo in mlp_topologies_under_over:
        print(f"Topologia oculta = {topo}")
        hist_examples = []
        for r in range(3):
            rng = np.random.RandomState(r + 7)
            Xtr, ytr, Xte, yte = train_test_split_random(X_norm, y_pm1, test_ratio=0.2, rng=rng)
            Xtr_T = Xtr.T; Ytr = ytr.reshape(1,-1)
            mlp = MultilayerPerceptron(Xtr_T, Ytr, topology=list(topo),
                                       learning_rate=hp['MLP']['lr'],
                                       tol=hp['MLP']['tol'], max_epoch=200)
            res = mlp.fit()
            hist_examples.append(res['eqm_history'])
        plt.figure(figsize=(6,4))
        for h in hist_examples:
            plt.plot(h, alpha=0.8)
        if topo[0] < 8:
            plt.title(f"Underfitting (topologia {topo})")
        elif topo[0] > 8:
            plt.title(f"Overfitting (topologia {topo})")
        else:
            plt.title(f"MLP (topologia {topo})")
        plt.xlabel('época'); plt.ylabel('EQM'); plt.grid(); plt.tight_layout(); plt.show()
        mlp_learning_histories[tuple(topo)] = hist_examples
    print("\nRESUMO FINAL (médias e desvios):")
    for m in ['Perceptron','ADALINE','MLP']:
        print(f"\n-- {m}")
        for mm in metrics_names:
            arr = np.array(results[m][mm])
            print(f" {mm:10s}: mean={arr.mean():.4f}, std={arr.std():.4f}, max={arr.max():.4f}, min={arr.min():.4f}")

    return {'results':results, 'confusions':confusions, 'mlp_histories':mlp_learning_histories}

if __name__ == "__main__":
    run_experiments(csv_path="spiral_d.csv", R=500, seed=123,
                          mlp_topologies_under_over=[[2],[8],[40]])
