import datetime
import pickle


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def save_checkpoint(path, *, step: int, model, optimizer) -> None:
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
