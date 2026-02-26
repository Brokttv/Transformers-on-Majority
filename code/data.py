
def data(seq_len:int, num_samples:int,Majority:bool):

  if Majority:
    X = np.random.randint(0,2,(num_samples,seq_len))
    y = (X.sum(axis=1) > seq_len //2).astype(int)
  else:
    X = np.zeros((num_samples,seq_len))
    X[0::2,:] = np.random.randint(0,2,(num_samples//2, seq_len))
    y = (X.sum(axis=1)>0).astype(int)

  return X,y
