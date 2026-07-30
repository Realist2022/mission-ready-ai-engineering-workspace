import hashlib

def dataset_hash(data):
    return hashlib.md5(str(data).encode()).hexdigest()

data_v1 = [1, 2, 3, 4, 5]
data_v2 = [1, 2, 3, 4, 6]  # small change

# Simple ML experiment versioning
def create_experiment_version(data, model_params, code_version):
    """Create a unique version ID for ML experiment"""
    experiment_info = {
        'data_hash': dataset_hash(data),
        'model_params': str(model_params),
        'code_version': code_version
    }
    version_string = str(experiment_info)
    return hashlib.md5(version_string.encode()).hexdigest()[:8]

# Example: Track different model experiments
model_v1 = {'learning_rate': 0.01, 'epochs': 10}
model_v2 = {'learning_rate': 0.001, 'epochs': 20}  # Different hyperparameters

exp_v1 = create_experiment_version(data_v1, model_v1, "code_v1.0")
exp_v2 = create_experiment_version(data_v1, model_v2, "code_v1.0")  # Same data, different model
exp_v3 = create_experiment_version(data_v2, model_v1, "code_v1.0")  # Different data, same model

print(f"\nExperiment 1 (lr=0.01): {exp_v1}")
print(f"Experiment 2 (lr=0.001): {exp_v2}")
print(f"Experiment 3 (new data): {exp_v3}")

# Show reproducibility
exp_v1_repeat = create_experiment_version(data_v1, model_v1, "code_v1.0")
print(f"\nReproducibility check:")
print(f"Original: {exp_v1}")
print(f"Repeated: {exp_v1_repeat}")
print(f"Same? {exp_v1 == exp_v1_repeat} âœ…")