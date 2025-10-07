import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import gc
import time
from tqdm import tqdm

from src.training import train_step, val_step

# Required external components: your data loading, train_step, test_step, and models
def train_with_mlflow(model: torch.nn.Module,
                      model_name: str,
                      train_dataloader,
                      val_dataloader,
                      loss_fn,
                      optimizer,
                      learning_rate,
                      epochs,
                      paths: dict):
    
    start_time = time.time()
    
    mlflow.log_param("model", model_name)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", type(optimizer).__name__)
    mlflow.log_param("loss_fn", type(loss_fn).__name__)
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    
    progress = tqdm(range(epochs),
                    desc=f"Training {model_name}",
                    dynamic_ncols=True,
                    leave=False)
    

    for epoch in progress:
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn)
        
        progress.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc*100:.2f}%",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc*100:.2f}%"
        })

        # Log metrics to MLFlow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # Save model
    total_time = time.time() - start_time
    mlflow.log_metric("train_time_sec", total_time)
    mlflow.set_tag("training_time_readable", f"{total_time:.2f} sec")
    
    model_path = paths["MODEL_CHECKPOINT_PATH"]  # raw models. They could be removed in the future. Mlflow doesn't use those checkpoints anyway
    torch.save(model.state_dict(), model_path)
        
    device = next(model.parameters()).device
    
    # Prepare input example and signature
    example_input, _ = next(iter(val_dataloader))
    example_input = example_input.to(device)
    example_output = model(example_input)
    signature = infer_signature(example_input.cpu().numpy(), example_output.detach().cpu().numpy())
    
    mlflow.pytorch.log_model(
        model, 
        name="model",
        input_example=example_input[:1].cpu().numpy(),
        signature=signature
    )

    return results