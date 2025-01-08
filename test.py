from utils import ConfigYAML
from dataset import get_dataloaders
from model import ModelSAM
from utils import setup_seed

if __name__ == "__main__":
    config = ConfigYAML("Testing configuration")
    args = config.parse()
    setup_seed(seed=42)
    train_loader, test_loader = get_dataloaders(args.DataConfig)
    model = ModelSAM(args)
    model.test(args.DataConfig.dataset, test_loader)
