"""
ANFIS - Adaptive Neuro-Fuzzy Inference System

5-layer neural-fuzzy architecture with hybrid learning.
Implements Gaussian membership functions and Takagi-Sugeno-Kang consequent.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class GaussianMF:
    """
    Gaussian membership function with learnable parameters.
    
    μ(x) = exp(-(x-c)² / 2σ²)
    """
    c: float  # Center
    sigma: float  # Width
    
    def forward(self, x: float) -> float:
        """Compute membership degree."""
        return np.exp(-((x - self.c) ** 2) / (2 * self.sigma ** 2))
    
    def gradient_c(self, x: float) -> float:
        """Gradient w.r.t. center: ∂μ/∂c = μ(x) · (x-c) / σ²"""
        mu = self.forward(x)
        return mu * (x - self.c) / (self.sigma ** 2)
    
    def gradient_sigma(self, x: float) -> float:
        """Gradient w.r.t. width: ∂μ/∂σ = μ(x) · (x-c)² / σ³"""
        mu = self.forward(x)
        return mu * ((x - self.c) ** 2) / (self.sigma ** 3)
    
    def to_dict(self) -> Dict:
        return {'c': self.c, 'sigma': self.sigma}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GaussianMF':
        return cls(c=d['c'], sigma=d['sigma'])


class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System.
    
    Architecture:
    Layer 1: Fuzzification (Gaussian MFs)
    Layer 2: Rule firing (product T-norm)
    Layer 3: Normalization
    Layer 4: Consequent (TSK first-order)
    Layer 5: Output summation
    """
    
    def __init__(self, n_inputs: int, n_rules: int, n_outputs: int = 1):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_outputs = n_outputs
        
        # Layer 1: Membership functions [n_inputs][n_rules]
        self.mfs: List[List[GaussianMF]] = []
        for i in range(n_inputs):
            input_mfs = []
            for j in range(n_rules):
                # Initialize with uniform coverage
                c = np.random.uniform(-1, 1)
                sigma = np.random.uniform(0.2, 0.8)
                input_mfs.append(GaussianMF(c, sigma))
            self.mfs.append(input_mfs)
        
        # Layer 4: Consequent parameters [n_rules][n_inputs + 1]
        # f_i(x) = a_i1*x1 + a_i2*x2 + ... + a_in*xn + c_i
        self.consequent_params = np.random.randn(n_rules, n_inputs + 1) * 0.1
        
        # Training history
        self.training_errors: List[float] = []
        self.validation_errors: List[float] = []
    
    def fuzzify(self, X: np.ndarray) -> np.ndarray:
        """
        Layer 1: Fuzzification
        
        Args:
            X: Input array [N, n_inputs]
        
        Returns:
            mu: Membership degrees [N, n_inputs, n_rules]
        """
        N = X.shape[0]
        mu = np.zeros((N, self.n_inputs, self.n_rules))
        
        for i in range(self.n_inputs):
            for j in range(self.n_rules):
                for n in range(N):
                    mu[n, i, j] = self.mfs[i][j].forward(X[n, i])
        
        return mu
    
    def compute_firing_strengths(self, mu: np.ndarray) -> np.ndarray:
        """
        Layer 2: Rule firing (product T-norm)
        
        Args:
            mu: Membership degrees [N, n_inputs, n_rules]
        
        Returns:
            w: Firing strengths [N, n_rules]
        """
        # Product of membership values for each rule
        return np.prod(mu, axis=1)
    
    def normalize(self, w: np.ndarray) -> np.ndarray:
        """
        Layer 3: Normalization
        
        Args:
            w: Firing strengths [N, n_rules]
        
        Returns:
            w_bar: Normalized firing strengths [N, n_rules]
        """
        w_sum = np.sum(w, axis=1, keepdims=True)
        return w / (w_sum + 1e-10)
    
    def compute_consequents(self, X: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
        """
        Layer 4: Consequent computation
        
        Args:
            X: Input array [N, n_inputs]
            w_bar: Normalized firing strengths [N, n_rules]
        
        Returns:
            O: Weighted consequent outputs [N, n_rules]
        """
        N = X.shape[0]
        O = np.zeros((N, self.n_rules))
        
        for n in range(N):
            for i in range(self.n_rules):
                # f_i(x) = a_i·x + c_i
                X_aug = np.append(X[n], 1.0)  # Add bias
                f_i = np.dot(self.consequent_params[i], X_aug)
                O[n, i] = w_bar[n, i] * f_i
        
        return O
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Complete forward pass through all 5 layers.
        
        Args:
            X: Input matrix [N, n_inputs]
        
        Returns:
            y: Output vector [N]
        """
        # Layer 1
        mu = self.fuzzify(X)
        
        # Layer 2
        w = self.compute_firing_strengths(mu)
        
        # Layer 3
        w_bar = self.normalize(w)
        
        # Layer 4
        O = self.compute_consequents(X, w_bar)
        
        # Layer 5: Output summation
        y = np.sum(O, axis=1)
        
        return y
    
    def least_squares_estimate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Forward pass: LSE for consequent parameters.
        
        Solves: θ_c = (A^T A)^(-1) A^T Y
        
        Args:
            X: Input [N, n_inputs]
            Y: Target [N]
        
        Returns:
            theta_c: Consequent parameters [n_rules, n_inputs + 1]
        """
        N = X.shape[0]
        
        # Compute firing strengths
        mu = self.fuzzify(X)
        w = self.compute_firing_strengths(mu)
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_bar = w / (w_sum + 1e-10)
        
        # Build A matrix [N, n_rules * (n_inputs + 1)]
        A = np.zeros((N, self.n_rules * (self.n_inputs + 1)))
        for n in range(N):
            row = []
            for i in range(self.n_rules):
                # w_bar_i * [x1, x2, ..., xn, 1]
                row.extend(w_bar[n, i] * np.append(X[n], 1.0))
            A[n] = row
        
        # LSE with regularization
        lambda_reg = 1e-6
        A_T_A = A.T @ A + lambda_reg * np.eye(A.shape[1])
        
        try:
            theta_c = np.linalg.solve(A_T_A, A.T @ Y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            theta_c = np.linalg.pinv(A) @ Y
        
        return theta_c.reshape(self.n_rules, self.n_inputs + 1)
    
    def gradient_descent_step(self, X: np.ndarray, Y: np.ndarray, 
                              learning_rate: float = 0.01) -> float:
        """
        Backward pass: GD for premise parameters.
        
        Updates c_i and sigma_i to minimize MSE.
        
        Args:
            X: Input [N, n_inputs]
            Y: Target [N]
            learning_rate: Step size
        
        Returns:
            mse: Mean squared error
        """
        N = X.shape[0]
        
        # Forward pass to get predictions
        y_pred = self.forward(X)
        error = Y - y_pred
        mse = np.mean(error ** 2)
        
        # Compute gradients for premise parameters
        # This is a simplified version - full implementation would use
        # proper chain rule through all layers
        
        for i in range(self.n_inputs):
            for j in range(self.n_rules):
                grad_c = 0.0
                grad_sigma = 0.0
                
                for n in range(N):
                    # Simplified gradient computation
                    x_val = X[n, i]
                    mf = self.mfs[i][j]
                    
                    grad_c += error[n] * mf.gradient_c(x_val)
                    grad_sigma += error[n] * mf.gradient_sigma(x_val)
                
                grad_c /= N
                grad_sigma /= N
                
                # Update parameters
                self.mfs[i][j].c -= learning_rate * grad_c
                self.mfs[i][j].sigma -= learning_rate * grad_sigma
                
                # Ensure sigma stays positive
                self.mfs[i][j].sigma = max(0.1, self.mfs[i][j].sigma)
        
        return mse
    
    def hybrid_learning_step(self, X: np.ndarray, Y: np.ndarray,
                            learning_rate: float = 0.01) -> Tuple[float, float]:
        """
        One epoch of hybrid learning.
        
        1. Forward pass: LSE for consequent parameters
        2. Backward pass: GD for premise parameters
        
        Returns:
            (train_mse, val_mse): Training and validation MSE
        """
        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        
        # Forward pass: Update consequent parameters with LSE
        self.consequent_params = self.least_squares_estimate(X_train, Y_train)
        
        # Backward pass: Update premise parameters with GD
        train_mse = self.gradient_descent_step(X_train, Y_train, learning_rate)
        
        # Validation
        y_val_pred = self.forward(X_val)
        val_mse = np.mean((Y_val - y_val_pred) ** 2)
        
        self.training_errors.append(train_mse)
        self.validation_errors.append(val_mse)
        
        return train_mse, val_mse
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100,
              learning_rate: float = 0.01, validation_split: float = 0.2,
              early_stopping_patience: int = 10, verbose: bool = True) -> Dict:
        """
        Complete training with hybrid learning.
        
        Args:
            X: Training inputs [N, n_inputs]
            Y: Training targets [N]
            epochs: Maximum training epochs
            learning_rate: Learning rate for GD
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before stopping
            verbose: Print progress
        
        Returns:
            history: Training history dictionary
        """
        # Normalize inputs
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-10
        X_norm = (X - self.X_mean) / self.X_std
        
        self.Y_mean = np.mean(Y)
        self.Y_std = np.std(Y) + 1e-10
        Y_norm = (Y - self.Y_mean) / self.Y_std
        
        best_val_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_mse, val_mse = self.hybrid_learning_step(
                X_norm, Y_norm, learning_rate
            )
            
            # Early stopping
            if val_mse < best_val_error:
                best_val_error = val_mse
                patience_counter = 0
                # Save best parameters
                self.best_mfs = [[GaussianMF(mf.c, mf.sigma) for mf in row] 
                                for row in self.mfs]
                self.best_consequent = self.consequent_params.copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    # Restore best parameters
                    self.mfs = self.best_mfs
                    self.consequent_params = self.best_consequent
                    break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train MSE = {train_mse:.6f}, "
                      f"Val MSE = {val_mse:.6f}")
        
        return {
            'epochs_trained': len(self.training_errors),
            'final_train_mse': self.training_errors[-1] if self.training_errors else None,
            'final_val_mse': self.validation_errors[-1] if self.validation_errors else None,
            'best_val_mse': best_val_error
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input [N, n_inputs]
        
        Returns:
            y: Predictions [N]
        """
        # Normalize
        X_norm = (X - self.X_mean) / self.X_std
        
        # Forward pass
        y_norm = self.forward(X_norm)
        
        # Denormalize
        y = y_norm * self.Y_std + self.Y_mean
        
        return y
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        y_pred = self.predict(X)
        
        mse = np.mean((Y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(Y - y_pred))
        
        # R-squared
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_fuzzy_rules(self) -> List[str]:
        """
        Extract human-readable fuzzy rules.
        
        Returns:
            rules: List of rule strings
        """
        rules = []
        
        for i in range(self.n_rules):
            # Antecedent
            antecedents = []
            for j in range(self.n_inputs):
                mf = self.mfs[j][i]
                antecedents.append(f"x{j+1} is Gaussian(c={mf.c:.2f}, σ={mf.sigma:.2f})")
            
            # Consequent
            params = self.consequent_params[i]
            consequent_terms = [f"{params[j]:.2f}*x{j+1}" for j in range(self.n_inputs)]
            consequent_terms.append(f"{params[-1]:.2f}")
            consequent = " + ".join(consequent_terms)
            
            rule = f"Rule {i+1}: IF {' AND '.join(antecedents)} THEN y = {consequent}"
            rules.append(rule)
        
        return rules
    
    def save(self, filepath: str):
        """Save model to file."""
        data = {
            'n_inputs': self.n_inputs,
            'n_rules': self.n_rules,
            'n_outputs': self.n_outputs,
            'mfs': [[mf.to_dict() for mf in row] for row in self.mfs],
            'consequent_params': self.consequent_params.tolist(),
            'X_mean': self.X_mean.tolist() if hasattr(self, 'X_mean') else None,
            'X_std': self.X_std.tolist() if hasattr(self, 'X_std') else None,
            'Y_mean': self.Y_mean if hasattr(self, 'Y_mean') else None,
            'Y_std': self.Y_std if hasattr(self, 'Y_std') else None,
            'training_errors': self.training_errors,
            'validation_errors': self.validation_errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ANFIS':
        """Load model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        model = cls(data['n_inputs'], data['n_rules'], data['n_outputs'])
        
        # Restore MFs
        model.mfs = [[GaussianMF.from_dict(mf_dict) for mf_dict in row] 
                     for row in data['mfs']]
        
        # Restore consequent params
        model.consequent_params = np.array(data['consequent_params'])
        
        # Restore normalization params
        if data['X_mean'] is not None:
            model.X_mean = np.array(data['X_mean'])
            model.X_std = np.array(data['X_std'])
            model.Y_mean = data['Y_mean']
            model.Y_std = data['Y_std']
        
        model.training_errors = data['training_errors']
        model.validation_errors = data['validation_errors']
        
        return model
