import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from NeuralNetWorks import *

def load_images_from_folder(root_folder, size=(40,40)):
    images, labels, class_names = [], [], []
    for idx, person in enumerate(sorted(os.listdir(root_folder))):
        person_path = os.path.join(root_folder, person)
        if not os.path.isdir(person_path):
            continue
        class_names.append(person)
        for fname in sorted(os.listdir(person_path)):
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_r = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            images.append(img_r.astype(np.float32))
            labels.append(idx)
    if len(images) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root_folder}.")
    images = np.stack(images, axis=0)   
    labels = np.array(labels, dtype=int)
    return images, labels, class_names

def build_X_Y(images, labels):
    N, H, W = images.shape
    p = H*W
    X_flat_T = images.reshape(N, p).T   
    C = labels.max() + 1
    Y = -np.ones((C, N), dtype=np.float32)
    for i in range(N):
        Y[labels[i], i] = 1.0
    return X_flat_T, Y

def normalize_X(X_T):
    mu = X_T.mean(axis=1, keepdims=True)
    sigma = X_T.std(axis=1, keepdims=True) + 1e-12
    return (X_T - mu)/sigma

def accuracy_score(y_true_idx, y_pred_idx):
    return np.mean(y_true_idx == y_pred_idx)

def confusion_matrix_multiclass(y_true_idx, y_pred_idx, C):
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[t, p] += 1
    return cm

def onehot_pm1_to_class_index(Y):
    return np.argmax(Y, axis=0)

def run_recfac(root_folder='recfac', img_size=(40,40), R=10, seed=123):
    hp = {
        'perceptron': {'lr':1e-3,'max_epochs':200},
        'adaline': {'lr':1e-3,'max_epoch':200,'tol':1e-6},
        'mlp': {'topology':[50],'lr':1e-3,'max_epoch':200,'tol':1e-4}
    }

    images, labels, class_names = load_images_from_folder(root_folder, size=img_size)
    N_total = images.shape[0]
    C = len(class_names)
    print(f"Lidas {N_total} imagens de {C} classes ({img_size[0]}x{img_size[1]})")

    X_p_N, Y_C_N = build_X_Y(images, labels)
    X_norm = normalize_X(X_p_N)
    y_indices = onehot_pm1_to_class_index(Y_C_N)

    models = ['Perceptron','ADALINE','MLP']
    accuracies = {m:[] for m in models}
    record = {m:{'best':None,'worst':None} for m in models}
    rng_global = np.random.RandomState(seed)

    for r in range(R):
        rng = np.random.RandomState(rng_global.randint(0,2**31-1))
        idx = rng.permutation(N_total)
        n_test = int(0.2 * N_total)
        test_idx = idx[:n_test]; train_idx = idx[n_test:]
        Xtr = X_norm[:, train_idx]; Xte = X_norm[:, test_idx]
        Ytr = Y_C_N[:, train_idx]; Yte = Y_C_N[:, test_idx]
        ytr_idx = y_indices[train_idx]; yte_idx = y_indices[test_idx]
        p = Xtr.shape[0]
        print(f"\n=== Rodada {r+1}/{R} ===")
        percs_w = np.zeros((C, p+1))
        for c in range(C):
            d_c = np.where(ytr_idx == c, 1, -1)
            perc = Perceptron(Xtr, d_c, learning_rate=hp['perceptron']['lr'], plot=False)
            perc.fit(max_epochs=hp['perceptron']['max_epochs'])
            percs_w[c, :] = perc.w.flatten()
        Xb_te = np.vstack((-np.ones((1, Xte.shape[1])), Xte))
        preds_idx_perc = np.argmax(percs_w @ Xb_te, axis=0)
        acc_perc = accuracy_score(yte_idx, preds_idx_perc)
        accuracies['Perceptron'].append(acc_perc)
        cm_perc = confusion_matrix_multiclass(yte_idx, preds_idx_perc, C)
        if record['Perceptron']['best'] is None or acc_perc > record['Perceptron']['best']['acc']:
            record['Perceptron']['best'] = {'acc':acc_perc,'cm':cm_perc}
        if record['Perceptron']['worst'] is None or acc_perc < record['Perceptron']['worst']['acc']:
            record['Perceptron']['worst'] = {'acc':acc_perc,'cm':cm_perc}
        adaline_w = np.zeros((C, p+1))
        for c in range(C):
            d_c = np.where(ytr_idx == c, 1.0, -1.0)
            ad = ADALINE(Xtr, d_c, learning_rate=hp['adaline']['lr'],
                         max_epoch=hp['adaline']['max_epoch'], tol=hp['adaline']['tol'], plot=False)
            ad.fit()
            adaline_w[c, :] = ad.w.flatten()
        preds_idx_ad = np.argmax(adaline_w @ Xb_te, axis=0)
        acc_ad = accuracy_score(yte_idx, preds_idx_ad)
        accuracies['ADALINE'].append(acc_ad)
        cm_ad = confusion_matrix_multiclass(yte_idx, preds_idx_ad, C)
        if record['ADALINE']['best'] is None or acc_ad > record['ADALINE']['best']['acc']:
            record['ADALINE']['best'] = {'acc':acc_ad,'cm':cm_ad}
        if record['ADALINE']['worst'] is None or acc_ad < record['ADALINE']['worst']['acc']:
            record['ADALINE']['worst'] = {'acc':acc_ad,'cm':cm_ad}
        mlp = MultilayerPerceptron(Xtr, Ytr, topology=hp['mlp']['topology'],
                                   learning_rate=hp['mlp']['lr'], tol=hp['mlp']['tol'],
                                   max_epoch=hp['mlp']['max_epoch'])
        mlp.fit()
        preds_mlp_idx = []
        for k in range(Xte.shape[1]):
            xk = Xte[:, [k]]
            out = mlp.predict(xk) 
            preds_mlp_idx.append(int(np.argmax(out[:,0])))
        preds_mlp_idx = np.array(preds_mlp_idx)
        acc_mlp = accuracy_score(yte_idx, preds_mlp_idx)
        accuracies['MLP'].append(acc_mlp)
        cm_mlp = confusion_matrix_multiclass(yte_idx, preds_mlp_idx, C)
        if record['MLP']['best'] is None or acc_mlp > record['MLP']['best']['acc']:
            record['MLP']['best'] = {'acc':acc_mlp,'cm':cm_mlp}
        if record['MLP']['worst'] is None or acc_mlp < record['MLP']['worst']['acc']:
            record['MLP']['worst'] = {'acc':acc_mlp,'cm':cm_mlp}

        print(f"Perceptron acc={acc_perc:.4f} | ADALINE acc={acc_ad:.4f} | MLP acc={acc_mlp:.4f}")
    summary = {}
    for m in models:
        arr = np.array(accuracies[m])
        summary[m] = {'mean':arr.mean(),'std':arr.std(),'max':arr.max(),'min':arr.min()}
    plt.figure(figsize=(7,5)); plt.boxplot([accuracies[m] for m in models], tick_labels=models); plt.title('Boxplot de acurácia'); plt.grid(); plt.tight_layout(); plt.show()
    for m in models:
        best = record[m]['best']; worst = record[m]['worst']
        fig, axs = plt.subplots(1,2,figsize=(12,5))
        sns.heatmap(best['cm'], annot=True, fmt='d', ax=axs[0], cmap='Blues')
        axs[0].set_title(f'{m} - MELHOR (acc={best["acc"]:.3f})')
        sns.heatmap(worst['cm'], annot=True, fmt='d', ax=axs[1], cmap='Reds')
        axs[1].set_title(f'{m} - PIOR (acc={worst["acc"]:.3f})')
        plt.suptitle(f'{m} - Matrizes (melhor vs pior)')
        plt.tight_layout(); plt.show()
    return {'accuracies':accuracies, 'record':record, 'summary':summary, 'class_names':class_names}

if __name__ == "__main__":
    models = ['Perceptron','ADALINE','MLP']
    OUT = run_recfac(root_folder='recfac', img_size=(40,40), R=10, seed=123)
    print("\nResumo Final (Acurácia):")
    for m,s in OUT['summary'].items():
        print(f"{m:12s} mean={s['mean']:.4f} std={s['std']:.4f} max={s['max']:.4f} min={s['min']:.4f}")
