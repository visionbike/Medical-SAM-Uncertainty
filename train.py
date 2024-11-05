from utils import ConfigYAML
from dataset import get_dataloaders
from model import ModelSAM

if __name__ == "__main__":
    config = ConfigYAML("Training configuration")
    args = config.parse()
    train_loader, test_loader = get_dataloaders(args.DataConfig)
    model = ModelSAM(args)
    model.train(args.DataConfig.dataset, train_loader, test_loader)
