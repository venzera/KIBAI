# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import catboost
import transformers
from transformers import AutoModel, EsmModel, AutoTokenizer
from tqdm import tqdm
import torch

# Suppress transformers warnings
transformers.logging.set_verbosity_error()

# Cell 1: Data Loading and Initial Processing
# Description: Load the dataset and filter for pH 7

# Load the dataset
df = pd.read_csv('tm.csv')

# Filter for pH 7
df = df[df['pH'] == 7]

# Cell 2: One-Hot Encoding Function
# Description: Define a function to create one-hot encoding for protein sequences

def getOneHot(seq):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_dict = {aa: 0 for aa in amino_acids}
    
    for aa in seq:
        amino_acid_dict[aa] += 1
    
    onehot = list(amino_acid_dict.values())
    return onehot

# Cell 3: Custom Train-Test Split Function
# Description: Define a function for stratified sampling in train-test split

def custom_train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Sort the data based on y values
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    step = n_samples // n_test
    
    # Select test indices at regular intervals
    test_indices = np.arange(0, n_samples, step)[:n_test]
    
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[test_indices] = True
    train_mask = ~test_mask
    
    X_test, y_test = X_sorted[test_mask], y_sorted[test_mask]
    X_train, y_train = X_sorted[train_mask], y_sorted[train_mask]
    
    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Cell 4: Feature Dictionaries
# Description: Define dictionaries for amino acid properties

# Hydrophobicity values for each amino acid
hydrophobicity = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
                  'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
                  'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

# Molecular weight for each amino acid
molecular_weight = {'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 'Q': 146.2, 'E': 147.1, 'G': 75.1,
                    'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1,
                    'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1}

# Charge for each amino acid
charge = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0,
          'H': 0, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
          'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# Cell 5: Enriched Encoding Function
# Description: Define a function to create enriched encoding for protein sequences

def getEnrichedEncoding(seq):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_count = {aa: seq.count(aa) for aa in amino_acids}
    seq_length = len(seq)
    aa_percentages = {aa: count / seq_length * 100 for aa, count in aa_count.items()}

    avg_hydrophobicity = sum(hydrophobicity[aa] * count for aa, count in aa_count.items()) / seq_length
    avg_molecular_weight = sum(molecular_weight[aa] * count for aa, count in aa_count.items()) / seq_length
    net_charge = sum(charge[aa] * count for aa, count in aa_count.items())
    
    encoding = (
        list(aa_count.values()) +
        list(aa_percentages.values()) +
        [seq_length, avg_hydrophobicity, avg_molecular_weight, net_charge]
    )

    return encoding

# Cell 6: Apply Encodings to DataFrame
# Description: Create new columns in the DataFrame for different encodings

# Apply one-hot encoding
df['onehot'] = df['protein_sequence'].apply(getOneHot)

# Apply enriched encoding
df['enriched_encoding'] = df['protein_sequence'].apply(getEnrichedEncoding)

# Cell 7: Model Training and Evaluation (One-Hot Encoding)
# Description: Train and evaluate a CatBoost model using one-hot encoding

X = df['onehot'].to_list()
y = df['tm'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = catboost.CatBoostRegressor(loss_function='RMSE', task_type='GPU', silent=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("One-Hot Encoding Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}")

# Cell 8: Model Training and Evaluation (Enriched Encoding)
# Description: Train and evaluate a CatBoost model using enriched encoding

X = df['enriched_encoding'].to_list()
y = df['tm'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = catboost.CatBoostRegressor(loss_function='RMSE', task_type='GPU', silent=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Enriched Encoding Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}")

# Cell 9: Prepare Data for ESM2 Model
# Description: Filter data based on sequence length for ESM2 model compatibility

df['len'] = df['protein_sequence'].apply(len)
df = df[(df['len'] < 1020) & (df['len'] > 50)]

# Cell 10: ESM2 Embedding Function
# Description: Define a function to create embeddings using the ESM2 model

def create_embedding(df):
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda")
    emb = []
    for i in tqdm(range(0, len(df))):
        inputs = tokenizer(df["protein_sequence"].iloc[i], return_tensors="pt", max_length=10000, truncation=True, padding=False).to("cuda")
        with torch.no_grad():
            emb.append(np.array(torch.mean(model(**inputs).last_hidden_state.cpu(), dim=1)))

    del model
    del tokenizer
    del df
    del inputs
    torch.cuda.empty_cache()
    return emb

# Cell 11: Create and Apply ESM2 Embeddings
# Description: Generate ESM2 embeddings for the protein sequences

embeddings = create_embedding(df)
embeddings = [emb.reshape(320) for emb in embeddings]
df['num_embs'] = embeddings

# Cell 12: Model Training and Evaluation (ESM2 Embeddings)
# Description: Train and evaluate a CatBoost model using ESM2 embeddings

X = df['num_embs'].to_list()
y = df['tm'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = catboost.CatBoostRegressor(loss_function='RMSE', task_type='GPU', silent=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("ESM2 Embedding Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}")

# Cell 13: Save Processed DataFrame
# Description: Save the processed DataFrame with all encodings and embeddings

df.to_pickle('df_esm_8M.pkl')
