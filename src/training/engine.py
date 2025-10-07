import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()  # item() uzyskuje wartości skalarne z tensora. Uzyskane wyniki nie muszą być na GPU. Pozwoli to uniknąć błędów przy matplotlibie
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # kumulacyjne accuracy (przez wszystkie batche)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc /len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # kumulacyjne accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def eval_step(model, test_dataloader):
    """
    Evaluates a single model on test dataset.
    
    Args:
        model: torch.nn.Module
        test_dataloader: DataLoader
    
    Returns:
        y_true: np.array of true labels
        y_probs: np.array of predicted probabilities
    """
    model = model.to(device).eval()
    probs_list = []
    labels_list = []

    with torch.inference_mode():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())
            labels_list.append(y_batch.cpu())

    y_true = torch.cat(labels_list).numpy()
    y_probs = torch.cat(probs_list).numpy()

    return y_true, y_probs

