from utils import ConfigYAML
from dataset import get_dataloaders
from model import ModelSAM
from utils import setup_seed

if __name__ == "__main__":
    config = ConfigYAML("Training configuration")
    args = config.parse()
    setup_seed(seed=42)
    train_loader, test_loader = get_dataloaders(args.DataConfig)
    model = ModelSAM(args)
    model.train(args.DataConfig.dataset, train_loader, test_loader)
