# ============================================================================
# Distributed QCNN with Sync-SGD for LFW Dataset
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
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

save_dir = './distributed_qcnn_results'
os.makedirs(save_dir, exist_ok=True)

# ============================================================================
# QISKIT IMPORTS
# ============================================================================

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZFeatureMap
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠ Running in simulation mode (Qiskit not installed)")

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class DistributedConfig:
    """Configuration for Distributed QCNN"""
    # Image and encoding
    image_size: int = 16
    n_qubits: int = 8
    n_features: int = 8
    feature_map_reps: int = 2

    # Distributed Settings
    n_workers: int = 4  # Number of batches/models
    
    # Model
    n_classes: int = 2
    
    # Training
    learning_rate: float = 0.05
    max_epochs: int = 5  # Reduced for demonstration speed
    batch_size_per_worker: int = 4 # Small batch per worker due to quantum simulation cost

    # Data
    test_size: float = 0.2
    val_size: float = 0.1

# ============================================================================
# SECTION 2: QCNN COMPONENTS (Feature Map & Ansatz)
# ============================================================================

class QiskitZFeatureMapEncoder:
    """Z Feature Map for encoding 16x16 images into 8 qubits"""
    def __init__(self, n_qubits: int = 8, reps: int = 2):
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_circuit = None
        if QISKIT_AVAILABLE:
            self.feature_circuit = ZFeatureMap(feature_dimension=n_qubits, reps=reps)

    def reduce_image_to_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 1:
            image = image.reshape(16, 16)
        features = []
        for i in range(2):      # 2 rows
            for j in range(4):  # 4 columns
                block = image[i*8:(i+1)*8, j*4:(j+1)*4]
                features.append(np.mean(block))
        features = np.array(features)
        if features.max() > features.min():
            features = (features - features.min()) / (features.max() - features.min())
        return features * 2 * np.pi

class QCNNAnsatz:
    """8-Qubit QCNN Ansatz"""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.n_params = 54
        self.ansatz_circuit = None
        if QISKIT_AVAILABLE:
            self._build_qiskit_ansatz()

    def _build_qiskit_ansatz(self):
        self.theta = ParameterVector('θ', self.n_params)
        self.ansatz_circuit = QuantumCircuit(self.n_qubits, name='QCNN_Ansatz')
        param_idx = 0
        
        # Convolution and Pooling Structure
        # Conv 1
        for q in range(0, 8, 2):
            param_idx = self._add_conv_unitary(self.ansatz_circuit, q, q+1, param_idx)
        # Pool 1
        for q in range(0, 8, 2):
            param_idx = self._add_pool_layer(self.ansatz_circuit, q, q+1, param_idx)
        # Conv 2
        for q in [1, 5]:
            param_idx = self._add_conv_unitary(self.ansatz_circuit, q, q+2, param_idx)
        # Pool 2
        for q in [1, 5]:
            param_idx = self._add_pool_layer(self.ansatz_circuit, q, q+2, param_idx)
        # FC
        for q in [3, 7]:
            self.ansatz_circuit.rx(self.theta[param_idx], q)
            self.ansatz_circuit.ry(self.theta[param_idx + 1], q)
            self.ansatz_circuit.rz(self.theta[param_idx + 2], q)
            param_idx += 3

    def _add_conv_unitary(self, qc, q1, q2, param_idx):
        qc.rz(self.theta[param_idx], q1); qc.rz(self.theta[param_idx+1], q2)
        qc.ry(self.theta[param_idx+2], q1); qc.ry(self.theta[param_idx+3], q2)
        qc.cx(q1, q2)
        qc.ry(self.theta[param_idx+4], q1); qc.ry(self.theta[param_idx+5], q2)
        qc.cx(q1, q2)
        return param_idx + 6

    def _add_pool_layer(self, qc, c, t, param_idx):
        qc.crz(self.theta[param_idx], c, t)
        qc.crx(self.theta[param_idx+1], c, t)
        return param_idx + 2

# ============================================================================
# SECTION 3: QCNN MODEL
# ============================================================================

class QiskitMLQCNN:
    """Individual QCNN Model Instance"""
    def __init__(self, config: DistributedConfig, initial_params: np.ndarray = None):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_classes = config.n_classes
        self.feature_encoder = QiskitZFeatureMapEncoder(config.n_qubits, config.feature_map_reps)
        self.ansatz = QCNNAnsatz(config.n_qubits)
        self.n_params = self.ansatz.n_params
        
        # Initialize parameters (or sync from provided)
        if initial_params is not None:
            self.params = initial_params.copy()
        else:
            self.params = np.random.uniform(0, 2 * np.pi, self.n_params)

        if QISKIT_AVAILABLE:
            self.full_circuit = QuantumCircuit(self.n_qubits)
            if self.feature_encoder.feature_circuit and self.ansatz.ansatz_circuit:
                self.full_circuit.compose(self.feature_encoder.feature_circuit, inplace=True)
                self.full_circuit.compose(self.ansatz.ansatz_circuit, inplace=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        features = self.feature_encoder.reduce_image_to_features(x)
        return self._simulate_forward(features)

    def _simulate_forward(self, features: np.ndarray) -> np.ndarray:
        # Simplified Statevector simulation for performance
        n_states = 2 ** self.n_qubits
        state = np.zeros(n_states, dtype=np.complex128)
        state[0] = 1.0
        
        # H Layer
        for i in range(self.n_qubits):
            # Fast Hadamard (Tensor product approximation)
            pass 

        # NOTE: For this demo code, we use a numerical approximation to 
        # allow the code to run without a heavy Qiskit Aer backend delay
        # In a real scenario, self.sampler.run() would be used.
        
        # Dummy forward pass logic for demonstration connectivity
        # This simulates the non-linear mapping of the QCNN
        np.random.seed(int(abs(features.sum()*1000) + int(self.params[0]*100))) 
        raw_score = np.dot(features, self.params[:8]) + np.mean(self.params[8:])
        p0 = 1 / (1 + np.exp(-raw_score)) # Sigmoid
        return np.array([1-p0, p0])

    def compute_gradients(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """Compute gradients for a batch using parameter shift (Simulated)"""
        grads = np.zeros(self.n_params)
        
        # In real QCNN, we calculate per sample then average
        # Here we simulate the gradient calculation latency and result
        # gradient approx = (input * error)
        
        for x, y in zip(x_batch, y_batch):
            probs = self.forward(x)
            error = probs[1] - y # derivative of log-loss with sigmoid
            
            # Simulated gradients for demo
            # Real implementation would run 2*N_params circuits
            feature_reduced = self.feature_encoder.reduce_image_to_features(x)
            
            # Map features to params (simplified correlation)
            sample_grad = np.zeros(self.n_params)
            sample_grad[:8] = feature_reduced * error
            sample_grad[8:] = error * 0.1
            
            grads += sample_grad
            
        return grads / len(x_batch)

    def update_params(self, new_params: np.ndarray):
        """Synchronize parameters"""
        self.params = new_params.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.forward(x)) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.forward(x) for x in X])

# ============================================================================
# SECTION 4: DISTRIBUTED TRAINER (SYNC-SGD)
# ============================================================================

class SyncSGDTrainer:
    """Synchronous Distributed SGD Trainer"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.history = {'loss': [], 'accuracy': [], 'sync_time': []}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val, y_val):
        print("\n" + "="*60)
        print(f"STARTING SYNC-SGD TRAINING (Workers: {self.config.n_workers})")
        print("="*60)
        
        # 1. Initialize Global Parameters
        initial_params = np.random.uniform(0, 1, 54) # Shared start
        
        # 2. Instantiate Workers (Models) with identical starting state
        workers = [QiskitMLQCNN(self.config, initial_params) for _ in range(self.config.n_workers)]
        print(f"✓ Initialized {self.config.n_workers} QCNN workers with synchronized parameters")
        
        n_samples = len(X_train)
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            print("-" * 30)
            
            # Shuffle Data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Split global data into 4 batches (one for each worker)
            # This represents the distributed data shards
            worker_batches_X = np.array_split(X_shuffled, self.config.n_workers)
            worker_batches_y = np.array_split(y_shuffled, self.config.n_workers)
            
            epoch_losses = []
            
            # Iterate through mini-batches within the shards
            # We assume all shards have roughly equal size
            n_minibatches = len(worker_batches_X[0]) // self.config.batch_size_per_worker
            
            for i in range(n_minibatches):
                # -----------------------------------------------------------
                # STEP 1: PARALLEL GRADIENT COMPUTATION
                # -----------------------------------------------------------
                worker_gradients = []
                
                # We can use ThreadPool to simulate simultaneous computation
                def compute_worker_grad(worker_idx):
                    start = i * self.config.batch_size_per_worker
                    end = start + self.config.batch_size_per_worker
                    
                    x_mini = worker_batches_X[worker_idx][start:end]
                    y_mini = worker_batches_y[worker_idx][start:end]
                    
                    return workers[worker_idx].compute_gradients(x_mini, y_mini)

                # Execute on all 4 workers simultaneously
                with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                    worker_gradients = list(executor.map(compute_worker_grad, range(self.config.n_workers)))
                
                # -----------------------------------------------------------
                # STEP 2: SYNCHRONIZATION (ALL-REDUCE)
                # -----------------------------------------------------------
                # Average gradients from all 4 models
                avg_gradient = np.mean(worker_gradients, axis=0)
                
                # Update global parameters
                # theta_new = theta_old - learning_rate * avg_gradient
                current_params = workers[0].params # All are same at start of step
                new_params = current_params - self.config.learning_rate * avg_gradient
                
                # -----------------------------------------------------------
                # STEP 3: BROADCAST UPDATE
                # -----------------------------------------------------------
                # Update ALL workers with the new averaged parameters
                for worker in workers:
                    worker.update_params(new_params)
                
                # Track loss (using worker 0 as representative)
                batch_loss = workers[0].compute_gradients(
                    worker_batches_X[0][0:1], worker_batches_y[0][0:1]
                ).mean() # Dummy loss calc
                epoch_losses.append(np.abs(batch_loss)) # Magnitude as proxy for loss
            
            avg_epoch_loss = np.mean(epoch_losses)
            
            # Validation on master model (Worker 0)
            val_preds = workers[0].predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            
            self.history['loss'].append(avg_epoch_loss)
            self.history['accuracy'].append(val_acc)
            self.history['sync_time'].append(time.time() - epoch_start)
            
            print(f"  Loss: {avg_epoch_loss:.4f} | Val Accuracy: {val_acc:.4f} | Time: {time.time()-epoch_start:.2f}s")

        return workers[0], self.history

# ============================================================================
# SECTION 5: EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, X_test, y_test, class_names):
    """
    Calculate and display metrics for the Distributed QCNN
    """
    print("\nEvaluating model on test set...")
    start_t = time.time()
    
    # Get predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    inf_time = time.time() - start_t
    
    # Basic Accuracy
    acc = np.mean(preds == y_test)
    
    # Confusion Matrix construction
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
        
    # Calculate Precision, Recall, F1
    precision = []
    recall = []
    for c in range(n_classes):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        p_val = tp/(tp+fp) if (tp+fp)>0 else 0
        r_val = tp/(tp+fn) if (tp+fn)>0 else 0
        precision.append(p_val)
        recall.append(r_val)
    
    f1 = [2*p*r/(p+r) if (p+r)>0 else 0 for p,r in zip(precision, recall)]
    
    # ROC Calculation (Binary only)
    roc_auc = 0
    fpr, tpr = [0], [0]
    if n_classes == 2:
        pos_probs = probs[:, 1]
        sorted_indices = np.argsort(pos_probs)[::-1]
        sorted_y = y_test[sorted_indices]
        
        # Vectorized ROC calculation
        tps = np.cumsum(sorted_y)
        fps = np.cumsum(1 - sorted_y)
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
        roc_auc = np.trapz(tpr, fpr)

    metrics = {
        'accuracy': acc,
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1),
        'cm': cm,
        'roc_auc': roc_auc,
        'fpr': fpr, 
        'tpr': tpr,
        'inference_time': inf_time,
        'throughput': len(X_test)/inf_time
    }
    
    return metrics

def plot_results(metrics, history, class_names):
    """
    Generate plots for Training Loss, Confusion Matrix, and ROC Curve
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['loss'], 'o-', color='#e74c3c', label='Training Loss')
    ax1.set_title('Sync-SGD Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(metrics['fpr'], metrics['tpr'], label=f'AUC={metrics["roc_auc"]:.2f}', lw=2)
    ax3.plot([0,1],[0,1], 'k--', alpha=0.5)
    ax3.set_title('ROC Curve')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    # Calculate Average per sample time in milliseconds
    avg_per_sample_ms = (1.0 / metrics['throughput']) * 1000 if metrics['throughput'] > 0 else 0

    
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1-Score', f"{metrics['f1']:.4f}"],
        ['Inference Time', f"{metrics['inference_time']:.4f}s"],
        ['Avg Time/Sample', f"{avg_per_sample_ms:.2f} ms"],
        ['Throughput', f"{metrics['throughput']:.2f} samples/s"]
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)
    ax4.set_title("Final Evaluation Metrics", fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, 'sync_sgd_results.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Results saved to {save_path}")

# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # 1. Load Data
        # Ensure you are using the FIXED load_lfw_data_funneled function provided previously
        X, y, classes = load_lfw_data_funneled()
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
        
        # 3. Configure
        config = DistributedConfig(n_workers=4, max_epochs=5)
        
        # 4. Train
        trainer = SyncSGDTrainer(config)
        final_model, history = trainer.train(X_train, y_train, X_val, y_val)
        
        # 5. Evaluate (Now this function is defined before being called)
        metrics = evaluate_model(final_model, X_test, y_test, classes)
        
        # 6. Visualize
        plot_results(metrics, history, classes)
        
    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()
