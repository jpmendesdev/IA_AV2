import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def ensure_2d_array(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1,1)
    return a

def add_bias(X):
    X = ensure_2d_array(X)
    return np.hstack([np.ones((X.shape[0],1)), X])

def train_test_split(X, y, test_size=0.2, rng=None):
    if rng is None: rng = np.random.default_rng()
    N = X.shape[0]
    idx = rng.permutation(N)
    cut = int(np.floor((1-test_size)*N))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normalize_standard(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std==0] = 1.0
    return (X-mean)/std, mean, std

# ---------- Metrics & confusion matrix ----------

def confusion_matrix_custom(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true,y_pred]))
    L = len(labels)
    lab_to_i = {lab:i for i,lab in enumerate(labels)}
    C = np.zeros((L,L), dtype=int)
    for t,p in zip(y_true, y_pred):
        C[lab_to_i[t], lab_to_i[p]] += 1
    return C, labels

def accuracy_from_confusion(C):
    return np.trace(C)/np.sum(C) if np.sum(C)>0 else 0.0

def precision_sensitivity_specificity_f1(C, pos_label_index=1):
    if C.shape != (2,2):
        raise ValueError("This helper expects a 2x2 confusion matrix for binary metrics.")
    TP = C[1,1]
    TN = C[0,0]
    FP = C[0,1]
    FN = C[1,0]
    acc = (TP+TN)/C.sum() if C.sum()>0 else 0.0
    sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0.0
    return {'accuracy':acc,'sensitivity':sens,'specificity':spec,'precision':prec,'f1':f1}

# ---------- Data loaders and processing ----------

def load_spiral_csv(path='spiral_d.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found. Please place spiral_d.csv in the working directory.")
    data = np.loadtxt(path, delimiter=',')
    if data.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: x,y,label")
    X = data[:,:2]
    y = data[:,2].astype(int)
    return X, y

# Função de redimensionamento mantida como fallback para PIL/plt.imread
def resize_image_numpy(img, new_h, new_w):
    h,w = img.shape
    row_scale = h / new_h
    col_scale = w / new_w
    out = np.zeros((new_h,new_w), dtype=float)
    for i in range(new_h):
        r0 = int(np.floor(i*row_scale))
        r1 = int(np.floor((i+1)*row_scale))
        if r1<=r0: r1 = r0+1
        for j in range(new_w):
            c0 = int(np.floor(j*col_scale))
            c1 = int(np.floor((j+1)*col_scale))
            if c1<=c0: c1 = c0+1
            block = img[r0:r1, c0:c1]
            out[i,j] = block.mean() if block.size>0 else 0.0
    return out

def load_recfac(folder='recfac', choose_size=(40,40)):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"A pasta '{folder}' não foi encontrada. Certifique-se de descompactar o RecFac.zip e nomear a pasta como '{folder}'.")

    valid_exts = ('.png', '.jpg', '.jpeg')
    images = []
    labels = []
    subject_map = {}
    next_label = 0

    print(f"Buscando imagens em subpastas dentro de '{folder}'...")

    for root, _, filenames in os.walk(folder):
        subject_name = os.path.basename(root)
        
        if not filenames or subject_name in (folder, ''):
            continue
            
        if subject_name not in subject_map:
            subject_map[subject_name] = next_label
            next_label += 1
        
        current_label = subject_map[subject_name]
        
        for filename in sorted(filenames):
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(root, filename)
                
                img_cv = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img_cv is None: raise IOError("CV2 falhou ao ler a imagem.")
                        
                # Redimensionamento otimizado com interpolação linear
                img_small = cv2.resize(img_cv, choose_size, interpolation=cv2.INTER_LINEAR)
                img_small = img_small.astype(float)
                    
                images.append(img_small.flatten())
                labels.append(current_label)

    if len(images) == 0:
        raise FileNotFoundError(f"Nenhum arquivo de imagem ({', '.join(valid_exts)}) encontrado dentro das subpastas de '{folder}'.")

    print(f"Total de imagens carregadas: {len(images)}")
    print(f"Total de classes (sujeitos) encontrados: {next_label}")

    X = np.vstack(images)
    y = np.array(labels, dtype=int)
    
    return X, y

def one_hot(y, C=None):
    y = np.asarray(y).astype(int)
    if C is None: C = np.max(y)+1
    out = -np.ones((len(y), C), dtype=float)
    for i,lab in enumerate(y):
        out[i,:] = -1.0; out[i,lab] = 1.0
    return out

def one_hot_standard(y, C=None):
    y = np.asarray(y).astype(int)
    if C is None: C = np.max(y)+1
    out = np.zeros((len(y), C), dtype=float)
    for i,lab in enumerate(y):
        out[i,lab] = 1.0
    return out

# ---------- Plotting helpers (Mantidas aqui pois são utilitárias) ----------

def plot_scatter(X, y, title='Scatter plot'):
    plt.figure(figsize=(6,5))
    labs = np.unique(y)
    for lab in labs:
        mask = (y==lab)
        plt.scatter(X[mask,0], X[mask,1], label=str(lab), s=8)
    plt.legend()
    plt.title(title)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.tight_layout()

def plot_confusion_matrix(C, labels=None, title='Confusion matrix'):
    plt.figure(figsize=(5,4))
    plt.imshow(C, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    if labels is None: labels = np.arange(C.shape[0])
    plt.xticks(np.arange(len(labels)), labels); plt.yticks(np.arange(len(labels)), labels)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, str(C[i,j]), horizontalalignment="center", color="white" if C[i,j]>C.max()/2 else "black")
    plt.tight_layout()

def plot_learning_curve(history_dict, title='Learning curve'):
    plt.figure(figsize=(6,4))
    for k,v in history_dict.items():
        if isinstance(v, list) and v:
             plt.plot(v, label=k)
    plt.title(title); plt.legend(); plt.xlabel('Epochs'); plt.tight_layout()

def summary_statistics(metric_values):
    arr = np.array(metric_values)
    return {'mean':arr.mean(), 'std':arr.std(ddof=0), 'max':arr.max(), 'min':arr.min()}


# ====================================================================
# CLASSES DE REDES NEURAIS
# ====================================================================

# ---------- Perceptron (binary) ----------

class Perceptron:
    def __init__(self, lr=0.01, epochs=100, bipolar=True, tol=1e-6, random_state=None):
        self.lr = lr
        self.epochs = epochs
        self.bipolar = bipolar
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.w = None
        self.history = {'loss':[], 'acc':[]}

    def activation(self, x):
        if self.bipolar:
            return np.where(x>=0, 1, -1)
        else:
            return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        X = ensure_2d_array(X); N, p = X.shape; Xb = add_bias(X)
        y_proc = np.array(y).copy()
        if self.bipolar: y_proc = np.where(y_proc==0, -1, y_proc)
        self.w = self.rng.normal(scale=0.1, size=(p+1,))
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(Xb, y_proc):
                out = np.dot(self.w, xi)
                y_pred = 1 if out>=0 else -1 if self.bipolar else (1 if out>=0 else 0)
                delta = (target - y_pred)
                if delta != 0:
                    self.w += self.lr * delta * xi
                    errors += 1
            y_pred_all = self.activation(Xb.dot(self.w))
            acc = np.mean(y_pred_all==y_proc)
            self.history['acc'].append(acc)
            self.history['loss'].append(errors/N)
            if errors==0: break
        return self

    def predict(self, X):
        Xb = add_bias(X); out = Xb.dot(self.w)
        return self.activation(out)

# ---------- ADALINE (binary) ----------

class Adaline:
    def __init__(self, lr=0.001, epochs=100, bipolar=True, tol=1e-6, random_state=None):
        self.lr = lr; self.epochs = epochs; self.bipolar = bipolar; self.tol = tol
        self.rng = np.random.default_rng(random_state); self.w = None; self.history = {'mse':[]}

    def activation(self, x):
        if self.bipolar: return np.where(x>=0, 1, -1)
        else: return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        X = ensure_2d_array(X); Xb = add_bias(X); N, p = X.shape
        y_proc = np.array(y).astype(float)
        if self.bipolar: y_proc = np.where(y_proc==0, -1, y_proc)
        self.w = self.rng.normal(scale=0.01, size=(p+1,))
        for epoch in range(self.epochs):
            net = Xb.dot(self.w); error = y_proc - net; mse = np.mean(error**2)
            self.history['mse'].append(mse)
            grad = -2 * Xb.T.dot(error) / N
            self.w = self.w - self.lr * grad
            if mse < self.tol: break
        return self

    def predict(self, X):
        Xb = add_bias(X); out = Xb.dot(self.w)
        return self.activation(out)


# ---------- MLP (multilayer perceptron) ----------

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(y): return y*(1-y)
def tanh(x): return np.tanh(x)
def dtanh(y): return 1 - y**2

class MLP:
    def __init__(self, layer_sizes, activation='tanh', lr=0.01, epochs=200, batch_size=None, tol=1e-6, random_state=None):
        self.layer_sizes = layer_sizes; self.activation = activation; self.lr = lr; self.epochs = epochs
        self.batch_size = batch_size; self.tol = tol; self.rng = np.random.default_rng(random_state)
        self.W = []; self.history = {'loss':[],'acc':[]}; self._init_weights()

    def _init_weights(self):
        self.W = []
        for i in range(len(self.layer_sizes)-1):
            in_dim = self.layer_sizes[i] + 1; out_dim = self.layer_sizes[i+1]
            limit = np.sqrt(6/(in_dim+out_dim))
            W = self.rng.uniform(-limit, limit, size=(in_dim, out_dim))
            self.W.append(W)

    def _forward(self, X):
        A = X; activations = [A]; nets = []
        for i,W in enumerate(self.W[:-1]):
            A_aug = add_bias(A); net = A_aug.dot(W); nets.append(net)
            A = tanh(net) if self.activation == 'tanh' else sigmoid(net); activations.append(A)
        A_aug = add_bias(A); net = A_aug.dot(self.W[-1]); nets.append(net)
        ex = np.exp(net - np.max(net, axis=1, keepdims=True))
        out = ex / np.sum(ex, axis=1, keepdims=True); activations.append(out)
        return activations, nets

    def fit(self, X, Y_onehot, verbose=False):
        X = ensure_2d_array(X); N = X.shape[0]; bs = N if self.batch_size is None else self.batch_size
        for epoch in range(self.epochs):
            perm = np.arange(N); self.rng.shuffle(perm); loss_epoch = 0.0
            for start in range(0, N, bs):
                batch_idx = perm[start:start+bs]; Xb = X[batch_idx]; Yb = Y_onehot[batch_idx]
                activations, nets = self._forward(Xb); out = activations[-1]
                loss = -np.mean(np.sum(Yb * np.log(out + 1e-12), axis=1)); loss_epoch += loss * len(batch_idx)
                delta = out - Yb; grads = []
                A_prev = add_bias(activations[-2]); gradW = A_prev.T.dot(delta) / len(batch_idx); grads.append(gradW)
                delta_prev = delta
                for l in range(len(self.W)-2, -1, -1):
                    W_no_bias = self.W[l+1][1:,...]; delta = delta_prev.dot(W_no_bias.T)
                    A_l = activations[l+1]
                    delta = delta * dtanh(A_l) if self.activation == 'tanh' else delta * dsigmoid(A_l)
                    A_prev = add_bias(activations[l]); gradWl = A_prev.T.dot(delta) / len(batch_idx); grads.insert(0, gradWl)
                    delta_prev = delta
                for i in range(len(self.W)): self.W[i] -= self.lr * grads[i]
            loss_epoch /= N
            activations, _ = self._forward(X); preds = np.argmax(activations[-1], axis=1)
            true = np.argmax(Y_onehot, axis=1); acc = np.mean(preds==true)
            self.history['loss'].append(loss_epoch); self.history['acc'].append(acc)
            if loss_epoch < self.tol: break
        return self

    def predict(self, X):
        X = ensure_2d_array(X); activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        activations, _ = self._forward(ensure_2d_array(X))
        return activations[-1]

# ---------- RBF Network ----------

def gaussian_rbf(x, c, s):
    diff = x[:,None,:] - c[None,:,:]
    dist2 = np.sum(diff**2, axis=2)
    return np.exp(-dist2/(2*(s**2)))

class RBFNetwork:
    def __init__(self, n_centers=10, gamma=None, sigma=None, random_state=None):
        self.n_centers = n_centers; self.gamma = gamma; self.sigma = sigma; self.random_state = random_state
        self.rng = np.random.default_rng(random_state); self.centers = None; self.sigma_val = None; self.W = None
        # Inicialização do histórico para compatibilidade com a pipeline (Correção do erro 'history')
        self.history = {'loss': [], 'acc': []} 

    def _choose_centers(self, X):
        idx = self.rng.choice(X.shape[0], size=self.n_centers, replace=False)
        return X[idx].copy()

    def fit(self, X, Y_onehot):
        X = ensure_2d_array(X); N, p = X.shape; k = self.n_centers
        self.centers = self._choose_centers(X)
        if self.sigma is None:
            dists = np.sqrt(((self.centers[:,None,:]-self.centers[None,:,:])**2).sum(axis=2))
            self.sigma_val = np.mean(dists);
            if self.sigma_val == 0: self.sigma_val = 1.0
        else: self.sigma_val = self.sigma
        Phi = gaussian_rbf(X, self.centers, self.sigma_val)
        Phi_b = np.hstack([np.ones((Phi.shape[0],1)), Phi])
        self.W = np.linalg.pinv(Phi_b).dot(Y_onehot)
        
        # Adicionar histórico mínimo
        train_preds = self.predict(X)
        train_true = np.argmax(Y_onehot, axis=1)
        train_acc = np.mean(train_preds == train_true)
        self.history['loss'].append(0.0)
        self.history['acc'].append(train_acc) 
        
        return self

    def predict_proba(self, X):
        X = ensure_2d_array(X); Phi = gaussian_rbf(X, self.centers, self.sigma_val)
        Phi_b = np.hstack([np.ones((Phi.shape[0],1)), Phi])
        out = Phi_b.dot(self.W)
        ex = np.exp(out - np.max(out, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# ---------- Experiment pipeline (Mantida aqui pois depende das classes) ----------

def run_binary_classification_mc(X, y, model_name='perceptron', R=500, test_size=0.2, rng_seed=0, **model_kwargs):
    rng = np.random.default_rng(rng_seed)
    results = []
    
    for r in range(R):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, rng=rng)
        X_train_n, mean, std = normalize_standard(X_train)
        X_test_n = (X_test-mean)/std
        
        m = None
        if model_name=='perceptron':
            m = Perceptron(**model_kwargs); m.fit(X_train_n, y_train); y_pred = m.predict(X_test_n)
        elif model_name=='adaline':
            m = Adaline(**model_kwargs); m.fit(X_train_n, y_train); y_pred = m.predict(X_test_n)
        elif model_name=='mlp':
            C = np.max(y)+1
            Y_train = one_hot_standard(y_train, C)
            m = MLP(layer_sizes=[X_train_n.shape[1]] + model_kwargs.get('hidden_sizes',[10]) + [C],
                    activation=model_kwargs.get('activation','tanh'), lr=model_kwargs.get('lr',0.01),
                    epochs=model_kwargs.get('epochs',100), batch_size=model_kwargs.get('batch_size', None),
                    random_state=rng.integers(1e9)); m.fit(X_train_n, Y_train); y_pred = m.predict(X_test_n)
        elif model_name=='rbf':
            C = np.max(y)+1
            Y_train = one_hot_standard(y_train, C)
            m = RBFNetwork(n_centers=model_kwargs.get('n_centers',10), random_state=rng.integers(1e9))
            m.fit(X_train_n, Y_train); y_pred = m.predict(X_test_n)
        else:
            raise ValueError("Unknown model_name")
            
        Cmat, labels = confusion_matrix_custom(y_test, y_pred)
        metrics = {}
        if Cmat.shape == (2,2):
            metrics = precision_sensitivity_specificity_f1(Cmat)
        metrics['accuracy'] = accuracy_from_confusion(Cmat)
        
        results.append({'confusion':Cmat, 'metrics':metrics, 'labels':labels, 'model':model_name, 'history': m.history})
        
    return results