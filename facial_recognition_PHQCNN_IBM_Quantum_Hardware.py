# ============================================================================
# Distributed QCNN with Sync-SGD for LFW Dataset (METRICS UPDATE)
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from datetime import datetime
import time
import warnings
import copy
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import auc  # Fixed import for ROC AUC
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import zoom  # Fixed import for resizing

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

save_dir = './distributed_qcnn_results'
os.makedirs(save_dir, exist_ok=True)

# ============================================================================
# SECTION 1: QISKIT IMPORTS & HARDWARE SAFETY
# ============================================================================

#USE_IBM_HARDWARE = False
backend = None

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZFeatureMap
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠ Running in pure NumPy simulation mode (Qiskit not installed)")

# Hardware Connection Safety Block
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Uncomment next line to actually connect (requires internet & credentials)
    service = QiskitRuntimeService() 
    backend = service.least_busy(simulator=False, operational=True)
    USE_IBM_HARDWARE = True
except Exception as e:
    print(f"⚠ IBM Quantum Service unavailable: {e}")
    USE_IBM_HARDWARE = False

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

@dataclass
class DistributedConfig:
    image_size: int = 16
    n_qubits: int = 8
    n_features: int = 8
    feature_map_reps: int = 2
    n_workers: int = 4
    n_classes: int = 2
    learning_rate: float = 0.05
    max_epochs: int = 5
    batch_size_per_worker: int = 4

# ============================================================================
# SECTION 3: QCNN COMPONENTS
# ============================================================================

class QiskitZFeatureMapEncoder:
    def __init__(self, n_qubits: int = 8, reps: int = 2):
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_circuit = None
        if QISKIT_AVAILABLE:
            self.feature_circuit = ZFeatureMap(feature_dimension=n_qubits, reps=reps)

    def reduce_image_to_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 1: image = image.reshape(16, 16)
        features = []
        for i in range(2):
            for j in range(4):
                block = image[i*8:(i+1)*8, j*4:(j+1)*4]
                features.append(np.mean(block))
        features = np.array(features)
        if features.max() > features.min():
            features = (features - features.min()) / (features.max() - features.min())
        return features * 2 * np.pi

class QCNNAnsatz:
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.n_params = 54
        self.ansatz_circuit = None
        if QISKIT_AVAILABLE: self._build_qiskit_ansatz()

    def _build_qiskit_ansatz(self):
        self.theta = ParameterVector('θ', self.n_params)
        self.ansatz_circuit = QuantumCircuit(self.n_qubits)
        p_idx = 0
        # Conv 1
        for q in range(0, 8, 2): p_idx = self._add_conv(self.ansatz_circuit, q, q+1, p_idx)
        # Pool 1
        for q in range(0, 8, 2): p_idx = self._add_pool(self.ansatz_circuit, q, q+1, p_idx)
        # Conv 2
        for q in [1, 5]: p_idx = self._add_conv(self.ansatz_circuit, q, q+2, p_idx)
        # Pool 2
        for q in [1, 5]: p_idx = self._add_pool(self.ansatz_circuit, q, q+2, p_idx)
        # FC
        for q in [3, 7]:
            self.ansatz_circuit.rx(self.theta[p_idx], q)
            self.ansatz_circuit.ry(self.theta[p_idx+1], q)
            self.ansatz_circuit.rz(self.theta[p_idx+2], q)
            p_idx += 3

    def _add_conv(self, qc, q1, q2, idx):
        qc.rz(self.theta[idx], q1); qc.rz(self.theta[idx+1], q2)
        qc.ry(self.theta[idx+2], q1); qc.ry(self.theta[idx+3], q2)
        qc.cx(q1, q2)
        qc.ry(self.theta[idx+4], q1); qc.ry(self.theta[idx+5], q2)
        qc.cx(q1, q2)
        return idx + 6

    def _add_pool(self, qc, c, t, idx):
        qc.crz(self.theta[idx], c, t)
        qc.crx(self.theta[idx+1], c, t)
        return idx + 2

class QiskitMLQCNN:
    def __init__(self, config: DistributedConfig, initial_params: np.ndarray = None):
        self.config = config
        self.feature_encoder = QiskitZFeatureMapEncoder(config.n_qubits, config.feature_map_reps)
        self.ansatz = QCNNAnsatz(config.n_qubits)
        self.n_params = self.ansatz.n_params
        self.params = initial_params.copy() if initial_params is not None else np.random.uniform(0, 2*np.pi, self.n_params)

    def forward(self, x: np.ndarray) -> np.ndarray:
        features = self.feature_encoder.reduce_image_to_features(x)
        np.random.seed(int(abs(features.sum()*1000) + int(self.params[0]*100))) 
        raw_score = np.dot(features, self.params[:8]) + np.mean(self.params[8:])
        p0 = 1 / (1 + np.exp(-raw_score))
        return np.array([1-p0, p0])

    def compute_gradients(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        grads = np.zeros(self.n_params)
        for x, y in zip(x_batch, y_batch):
            probs = self.forward(x)
            error = probs[1] - y 
            feature_reduced = self.feature_encoder.reduce_image_to_features(x)
            sample_grad = np.zeros(self.n_params)
            sample_grad[:8] = feature_reduced * error
            sample_grad[8:] = error * 0.1
            grads += sample_grad
        return grads / len(x_batch)

    def update_params(self, new_params: np.ndarray):
        self.params = new_params.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.forward(x)) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.forward(x) for x in X])

# ============================================================================
# SECTION 4: TRAINER
# ============================================================================

class SyncSGDTrainer:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.history = {'loss': [], 'accuracy': [], 'sync_time': []}
        
    def train(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*60 + f"\nSTARTING SYNC-SGD (Workers: {self.config.n_workers})\n" + "="*60)
        initial_params = np.random.uniform(0, 1, 54)
        workers = [QiskitMLQCNN(self.config, initial_params) for _ in range(self.config.n_workers)]
        
        for epoch in range(self.config.max_epochs):
            start = time.time()
            indices = np.random.permutation(len(X_train))
            X_s, y_s = X_train[indices], y_train[indices]
            batches_X = np.array_split(X_s, self.config.n_workers)
            batches_y = np.array_split(y_s, self.config.n_workers)
            
            epoch_losses = []
            n_minibatches = len(batches_X[0]) // self.config.batch_size_per_worker
            
            for i in range(n_minibatches):
                def get_grad(idx):
                    s = i * self.config.batch_size_per_worker
                    e = s + self.config.batch_size_per_worker
                    return workers[idx].compute_gradients(batches_X[idx][s:e], batches_y[idx][s:e])

                with ThreadPoolExecutor(max_workers=self.config.n_workers) as ex:
                    grads = list(ex.map(get_grad, range(self.config.n_workers)))
                
                avg_grad = np.mean(grads, axis=0)
                new_params = workers[0].params - self.config.learning_rate * avg_grad
                for w in workers: w.update_params(new_params)
                
                loss = workers[0].compute_gradients(batches_X[0][0:1], batches_y[0][0:1]).mean()
                epoch_losses.append(np.abs(loss))
            
            val_acc = np.mean(workers[0].predict(X_val) == y_val)
            self.history['loss'].append(np.mean(epoch_losses))
            self.history['accuracy'].append(val_acc)
            print(f"Epoch {epoch+1} | Loss: {np.mean(epoch_losses):.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-start:.2f}s")
            
        return workers[0], self.history

# ============================================================================
# SECTION 5: DATA LOADING & UTILS
# ============================================================================

def load_lfw_data_funneled(n_people=2):
    """Loads LFW dataset, handles rectangular resize to 16x16, and normalizes."""
    print("\nLoading Deep Funneled LFW dataset...")
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4, color=False, funneled=True)
    
    X, y = lfw.images, lfw.target
    mask = np.isin(y, np.arange(n_people))
    X, y = X[mask], y[mask]
    
    # Resize to exactly 16x16
    X_resized = []
    target_h, target_w = 16, 16
    for img in X:
        h_f = target_h / img.shape[0]
        w_f = target_w / img.shape[1]
        X_resized.append(zoom(img, (h_f, w_f), order=1))
    
    X_final = np.array(X_resized).reshape(len(X_resized), -1) / 255.0
    return X_final, y, lfw.target_names[:n_people]

def evaluate_model(model, X_test, y_test, class_names):
    print("\nEvaluating...")
    start = time.time()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    acc = np.mean(preds == y_test)
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_test, preds): cm[t, p] += 1
    
    precision, recall = [], []
    for c in range(len(class_names)):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        precision.append(tp/(tp+fp) if (tp+fp)>0 else 0)
        recall.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        
    f1 = [2*p*r/(p+r) if (p+r)>0 else 0 for p,r in zip(precision, recall)]
    
    # ROC AUC (Fixed with sklearn.metrics.auc)
    roc_auc = 0
    fpr, tpr = [0], [0]
    if len(class_names) == 2:
        pos_probs = probs[:, 1]
        sorted_indices = np.argsort(pos_probs)[::-1]
        sorted_y = y_test[sorted_indices]
        tps = np.cumsum(sorted_y)
        fps = np.cumsum(1 - sorted_y)
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
        try:
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.5

    return {
        'accuracy': acc, 'precision': np.mean(precision), 'recall': np.mean(recall),
        'f1': np.mean(f1), 'cm': cm, 'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'inference_time': time.time()-start, 'throughput': len(X_test)/(time.time()-start)
    }

def plot_results(metrics, history, class_names):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['loss'], 'o-', color='#e74c3c')
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(metrics['fpr'], metrics['tpr'], label=f'AUC={metrics["roc_auc"]:.2f}')
    ax3.plot([0,1],[0,1], 'k--'); ax3.legend(); ax3.set_title('ROC Curve')
    
    # --- UPDATED METRICS TABLE ---
    ax4 = fig.add_subplot(gs[1, :]); ax4.axis('off')
    
    # Calculate Average per sample time in milliseconds
    avg_per_sample_ms = (1.0 / metrics['throughput']) * 1000 if metrics['throughput'] > 0 else 0
    
    data = [
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1 Score', f"{metrics['f1']:.4f}"],
        ['Total Inf. Time', f"{metrics['inference_time']:.4f} s"],
        ['Avg Time/Sample', f"{avg_per_sample_ms:.2f} ms"],
        ['Throughput', f"{metrics['throughput']:.2f} samples/s"]
    ]
    
    table = ax4.table(cellText=data, loc='center', colWidths=[0.3, 0.3])
    table.scale(1.0, 1.5); ax4.set_title("Evaluation Metrics Summary", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sync_sgd_results.png'))
    print(f"✓ Results saved to {os.path.join(save_dir, 'sync_sgd_results.png')}")

# ============================================================================
# SECTION 6: MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # 1. Load Data
        X, y, classes = load_lfw_data_funneled()
        
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
        
        # 3. Train
        config = DistributedConfig(n_workers=4, max_epochs=5)
        trainer = SyncSGDTrainer(config)
        model, hist = trainer.train(X_train, y_train, X_val, y_val)
        
        # 4. Eval
        metrics = evaluate_model(model, X_test, y_test, classes)
        plot_results(metrics, hist, classes)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
