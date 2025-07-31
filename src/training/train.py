import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from datetime import datetime
import torch
import time
from tqdm import tqdm

from src.training import train_step, val_step
from src.evaluation import save_loss_plot
from src.utils.paths import MLFLOW_TRACKING_DIR, CHECKPOINTS_DIR, FIGURES_DIR

mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())

# Required external components: your data loading, train_step, test_step, and models
def train_with_mlflow(model: torch.nn.Module,
                      model_name: str,
                      train_dataloader,
                      val_dataloader,
                      loss_fn,
                      optimizer,
                      learning_rate,
                      epochs,
                      paths: dict,
                      run_name=None):  # for future flexibility (if I would like add different naming for experiments in future)
    run_name = run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # then I would have to remove that line
    
    start_time = time.time()
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", type(optimizer).__name__)
        mlflow.log_param("loss_fn", type(loss_fn).__name__)
        
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        
        progress = tqdm(range(epochs),
                        desc=f"Training {model_name}",
                        dynamic_ncols=True,
                        leave=False)
        

        for epoch in progress:
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
            test_loss, test_acc = val_step(model, val_dataloader, loss_fn)
            
            progress.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc*100:.2f}%",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc*100:.2f}%"
            })

            # Log metrics to MLFlow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # Save model
        total_time = time.time() - start_time
        mlflow.log_metric("train_time_sec", total_time)
        mlflow.set_tag("training_time_readable", f"{total_time:.2f} sec")
        
        model_path = paths["MODEL_CHECKPOINT_PATH"]
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))
        
        device = next(model.parameters()).device
        
        # Prepare input example and signature
        example_input, _ = next(iter(val_dataloader))
        example_input = example_input.to(device)
        example_output = model(example_input)
        signature = infer_signature(example_input.cpu().numpy(), example_output.detach().cpu().numpy())
        
        mlflow.pytorch.log_model(
            model, 
            name="models",
            input_example=example_input[:1].cpu().numpy(),
            signature=signature
        )

        # Save training curve as image
        loss_curve_path = save_loss_plot(results, model_name, output_dir=paths["LOSS_PLOT_PATH"].parent)
        mlflow.log_artifact(str(loss_curve_path))

        return results