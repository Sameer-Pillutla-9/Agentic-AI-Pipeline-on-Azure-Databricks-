from .data_agent import DataAgent
from .feature_agent import FeatureAgent
from .model_agent import ModelAgent


def main() -> None:
    DataAgent().run()
    FeatureAgent().run()
    ModelAgent().run()


if __name__ == "__main__":
    main()
