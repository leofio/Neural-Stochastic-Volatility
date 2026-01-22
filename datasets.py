from torch.utils.data import Dataset, DataLoader

class options_dataset(Dataset):
  def __init__(self, df, device='cpu'):
    self.df  = df
    self.t = torch.tensor(df['Time'].values, dtype=torch.float32).unsqueeze(1).to(device)
    self.k = torch.tensor(df['Strike'].values, dtype=torch.float32).unsqueeze(1).to(device)
    self.v = torch.tensor(df['Volatility'].values, dtype=torch.float32).unsqueeze(1).to(device)
    self.x = torch.cat((self.k, self.t, self.v), dim=1)
    self.price = torch.tensor(df['Price'].values, dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    return self.x[idx], self.price[idx]
