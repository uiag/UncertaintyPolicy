#Contains the learned policy models
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch
import torch.nn as nn
import os
import joblib

#Simple Neural Net with 3 hidden layers of dimension 128
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, hidden//4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.head_correct = nn.Linear(hidden//4, 1)
        self.head_fix = nn.Linear(hidden//4, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head_correct(h).squeeze(-1), self.head_fix(h).squeeze(-1)

#Function to preprocess data (only choose feature columns and remove NaN/None rows)
def prepare_X(df, feature_cols):
    X = df[feature_cols].copy()
    X = X.dropna()
    return X.values.astype(float)

#Main class to handle all policy models
class PredictorSuite:
    def __init__(self, feature_cols, seed):
        self.feature_cols = feature_cols
        self.seed = seed

        #Linear model (two models)
        self.linear_corr = None
        self.linear_fix = None

        #LightGBM model (two models)
        self.lgbm_corr = None
        self.lgbm_fix = None

        #Neural Network model (one combined model)
        self.mlp_model = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Fit Ridge Regression to training data
    def fit_linear(self, df_train, output_dir):
        Xtr = prepare_X(df_train, self.feature_cols)
        ytr = df_train['p_corr_empirical']
        ytr_fix = df_train['p_fix_empirical']

        self.linear_corr = Ridge(alpha=0.1).fit(Xtr, ytr)
        self.linear_fix = Ridge(alpha=0.1).fit(Xtr, ytr_fix)

        #Save models
        joblib.dump(self.linear_corr, os.path.join(output_dir,f'linear_corr.joblib'))
        joblib.dump(self.linear_fix, os.path.join(output_dir,f'linear_fix.joblib'))

    #Predict on data using the linear model
    def predict_linear(self, df):
        X = prepare_X(df, self.feature_cols)
        p_corr = self.linear_corr.predict(X)
        p_fix = self.linear_fix.predict(X)
        return p_corr, p_fix

    #Fit LightGBM model to training data and apply early stopping on validation data
    def fit_lightgbm(self, df_train, df_val, output_dir):
        Xtr = prepare_X(df_train, self.feature_cols)
        ytr = df_train['p_corr_empirical']
        Xv = prepare_X(df_val, self.feature_cols)
        yv = df_val['p_corr_empirical']

        #Use bagging and feature fraction to counter overfitting
        params = {
            'objective':'regression', 'seed': self.seed, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'bagging_seed': self.seed, 'feature_fraction': 0.8, 'feature_fraction_seed': self.seed, 'early_stopping_round': 10, 'verbosity': 0
            }

        #Fit first model
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalidation = lgb.Dataset(Xv, label=yv)
        self.lgbm_corr = lgb.train(params, dtrain, valid_sets=[dvalidation], num_boost_round=100)

        #Fit second model
        ytr_fix = df_train['p_fix_empirical']
        yval_fix = df_val['p_fix_empirical']
        dtrain2 = lgb.Dataset(Xtr, label=ytr_fix)
        dvalidation2 = lgb.Dataset(Xv, label=yval_fix)
        self.lgbm_fix = lgb.train(params, dtrain2, valid_sets=[dvalidation2], num_boost_round=100)

        #Save models
        joblib.dump(self.lgbm_corr, os.path.join(output_dir,f'lgbm_corr.joblib'))
        joblib.dump(self.lgbm_fix, os.path.join(output_dir,f'lgbm_fix.joblib'))

    #Predict on data using the LightGBM model
    def predict_lightgbm(self, df):
        X = prepare_X(df, self.feature_cols)
        p_corr = self.lgbm_corr.predict(X)
        p_fix = self.lgbm_fix.predict(X)
        return p_corr, p_fix

    #Fit Neural Network model to training data and apply early stopping on validation data
    def fit_mlp(self, df_train, df_val, output_dir, patience=10, epochs=100, batch=512):
        Xtr = prepare_X(df_train, self.feature_cols)
        ytr = df_train['p_corr_empirical']
        ytr_fix = df_train['p_fix_empirical']
        Xv = prepare_X(df_val, self.feature_cols)
        yv = df_val['p_corr_empirical']
        yv_fix = df_val['p_fix_empirical']

        #Initialize model
        input_dim = Xtr.shape[1]
        model = MLP(input_dim).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), amsgrad=True) #Use AdamW with amsgrad
        loss_fn = nn.MSELoss() #Use MSE as loss

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(self.device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).to(self.device)
        ytr_fix_t = torch.tensor(ytr_fix, dtype=torch.float32).to(self.device)
        Xv_t = torch.tensor(Xv, dtype=torch.float32).to(self.device)
        yv_t = torch.tensor(yv, dtype=torch.float32).to(self.device)
        yv_fix_t = torch.tensor(yv_fix, dtype=torch.float32).to(self.device)
        N = Xtr_t.size(0)

        best_val_loss = float('inf')
        best_model = None
        counter = 0

        #Train for epochs
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            perm = torch.randperm(N)

            #Train using batches
            for i in range(0, N, batch):
                idx = perm[i:i+batch]
                xb = Xtr_t[idx]
                yc, yf = model(xb)
                loss = loss_fn(yc, ytr_t[idx]) + loss_fn(yf, ytr_fix_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            model.eval() #set model to evaluation mode
            with torch.no_grad():
                yc, yf = model(Xv_t)
                val_loss = loss_fn(yc, yv_t) + loss_fn(yf, yv_fix_t)

            #Track best model
            if val_loss < best_val_loss * 0.999:
                best_val_loss = val_loss
                best_model = model
                counter = 0

            #If model not better than best model increase stopping counter
            else:
                counter += 1
                if counter >= patience: #If patience is reached apply early stopping and stop training and save model
                    self.mlp_model = best_model
                    torch.save(model.state_dict(), os.path.join(output_dir, f'mlp.pt'))
                    return True

        #Save model
        self.mlp_model = model
        torch.save(model.state_dict(), os.path.join(output_dir,f'mlp.pt'))
        return True

    #Predict on data using the Neural Network model
    def predict_mlp(self, df):
        X = prepare_X(df, self.feature_cols)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.mlp_model.eval()
        with torch.no_grad():
            pc, pf = self.mlp_model(X_t)
        return pc.cpu().numpy(), pf.cpu().numpy()
